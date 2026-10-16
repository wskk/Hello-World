#!/usr/bin/env python3
"""运行 SGD / SGD+动量 / Adam 对照实验，并保存逐步数值和损失曲线。"""

import argparse
import csv
from dataclasses import asdict, dataclass
from html import escape
import json
import math
from pathlib import Path
import random
from typing import Tuple

from optimizers import Adam, Momentum, SGD


METHODS = ("SGD", "Momentum", "Adam")
GRID = (0.001, 0.003, 0.01, 0.018, 0.03, 0.1, 0.3, 1.0)
# 所有数据写在这里，完全无需下载；真实关系为 y = 2*x + 1。
DATA = tuple((float(x), float(2 * x + 1)) for x in (-3, -2, -1, 1, 2, 3))


@dataclass(frozen=True)
class Problem:
    name: str
    title: str
    names: Tuple[str, ...]
    start: Tuple[float, ...]
    default_steps: int
    default_lr: float

    def loss(self, parameters, batch=()):
        if self.name == "scalar":
            return 0.5 * (parameters[0] - 3) ** 2
        if self.name == "anisotropic":
            x, y = parameters
            return 0.5 * (x * x + 100 * y * y)
        ids = batch or tuple(range(len(DATA)))
        w, b = parameters
        return sum((w * DATA[i][0] + b - DATA[i][1]) ** 2 for i in ids) / (2 * len(ids))

    def gradient(self, parameters, batch=()):
        if self.name == "scalar":
            return [parameters[0] - 3]
        if self.name == "anisotropic":
            return [parameters[0], 100 * parameters[1]]
        ids = batch or tuple(range(len(DATA)))
        w, b = parameters
        # 前向：prediction = w*x+b；反向链式法则：dL/dw = mean(error*x)。
        errors = [(w * DATA[i][0] + b - DATA[i][1], DATA[i][0]) for i in ids]
        return [sum(error * x for error, x in errors) / len(ids),
                sum(error for error, _ in errors) / len(ids)]


PROBLEMS = {
    "scalar": Problem("scalar", "一维抛物线：L(w)=(w-3)^2/2", ("w",), (0.0,), 20, 0.1),
    "anisotropic": Problem("anisotropic", "二维尺度：L(x,y)=(x^2+100*y^2)/2",
                           ("x", "y"), (3.0, 3.0), 100, 0.01),
    "linear": Problem("linear", "小批量线性回归：prediction=w*x+b",
                      ("w", "b"), (0.0, 0.0), 60, 0.1),
}


def batch_schedule(problem, steps, seed):
    """预先生成一次批次顺序，所有优化器和学习率试验重复使用该顺序。"""
    if problem.name != "linear":
        return [()] * steps
    rng = random.Random(seed)
    batches = []
    while len(batches) < steps:
        ids = list(range(len(DATA)))
        rng.shuffle(ids)
        batches.extend(tuple(ids[i:i + 2]) for i in range(0, len(ids), 2))
    return batches[:steps]


def run(problem, method, lr, schedule, beta1=0.9, beta2=0.999, mu=0.9):
    """每次 run 都重建参数和状态；训练中梯度由当前位置重新计算。"""
    optimizer = {"SGD": lambda: SGD(lr), "Momentum": lambda: Momentum(lr, mu),
                 "Adam": lambda: Adam(lr, beta1, beta2)}[method]()
    parameters = list(problem.start)
    rows, losses = [], [problem.loss(parameters)]  # 曲线包括未更新的 t=0
    status = "completed"
    for step, batch in enumerate(schedule, 1):
        try:
            before_loss = problem.loss(parameters)
            batch_before = problem.loss(parameters, batch)
            gradients = problem.gradient(parameters, batch)
            updates = optimizer.step(parameters, gradients)
            new_parameters = [u.parameter_after for u in updates]
            after_loss = problem.loss(new_parameters)
            batch_after = problem.loss(new_parameters, batch)
            # 停止非有限数值的运行，不能把其上一步损失误报为最终成功结果。
            values = [before_loss, batch_before, after_loss, batch_after]
            values += [v for u in updates for v in asdict(u).values() if v is not None]
            if not all(math.isfinite(v) for v in values):
                raise OverflowError("non-finite update")
        except (OverflowError, ValueError):
            status = "nonfinite"
            break
        for name, update in zip(problem.names, updates):
            row = dict(problem=problem.name, optimizer=method, step=step,
                       epoch=(step - 1) // 3 + 1 if problem.name == "linear" else "",
                       batch_ids=" ".join(map(str, batch)), parameter=name, lr=lr,
                       loss_before=before_loss, loss_after=after_loss,
                       batch_loss_before=batch_before, batch_loss_after=batch_after)
            row.update(asdict(update))
            rows.append(row)
        parameters = new_parameters
        losses.append(after_loss)
    return dict(problem=problem.name, optimizer=method, lr=lr, status=status,
                requested_steps=len(schedule), completed_steps=len(losses) - 1,
                initial_loss=losses[0], final_loss=losses[-1] if status == "completed" else math.inf,
                best_loss=min(losses), parameters=parameters, rows=rows, losses=losses)


def number(value):
    if value is None or value == "":
        return "-"
    return f"{value:.9g}" if isinstance(value, float) else str(value)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)


def write_loss_svg(path, problem, runs):
    """标准库生成精确的 SVG 折线图，无图形库依赖。"""
    width, height = 860, 440
    left, top, plot_width, plot_height = 90, 70, 720, 285
    floor = 1e-12  # 仅用于对数作图，CSV 保留真实损失（包括 0）
    values = [math.log10(max(floor, loss)) for r in runs for loss in r["losses"]]
    low, high = math.floor(min(values)), math.ceil(max(values))
    if low == high:
        high += 1
    max_steps = max(r["requested_steps"] for r in runs)
    def position(step, loss):
        x = left + step / max_steps * plot_width
        y = top + (high - math.log10(max(floor, loss))) / (high - low) * plot_height
        return f"{x:.2f},{y:.2f}"
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<g font-family="Arial, sans-serif" font-size="13" fill="#243247">',
           f'<text x="{left}" y="28" font-size="18">{escape(problem.name)}: full-objective loss</text>',
           f'<text x="{left}" y="49">Equal lr={runs[0]["lr"]:g}; lower is better; logarithmic y-axis</text>']
    ticks = sorted(set(round(low + i * (high - low) / 5) for i in range(6)))
    for exponent in ticks:
        y = top + (high - exponent) / (high - low) * plot_height
        svg += [f'<path d="M{left} {y:.2f}h{plot_width}" stroke="#e1e7ef"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">1e{exponent}</text>']
    for step in sorted(set(round(i * max_steps / 5) for i in range(6))):
        x = left + step / max_steps * plot_width
        svg.append(f'<text x="{x:.2f}" y="380" text-anchor="middle">{step}</text>')
    colors = ("#2463c5", "#d27611", "#16875c")
    for i, (result, color) in enumerate(zip(runs, colors)):
        points = " ".join(position(t, loss) for t, loss in enumerate(result["losses"]))
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>')
        label = result["optimizer"] + (" (nonfinite; stopped)" if result["status"] != "completed" else "")
        svg.append(f'<text x="{left + i * 240}" y="408" fill="{color}">{escape(label)}</text>')
    svg += ['<text x="450" y="395" text-anchor="middle">step</text>',
            '<text x="90" y="432" font-size="11">Losses below 1e-12 are plotted at 1e-12; see CSV for exact values.</text>',
            '</g></svg>']
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def print_trace(problem, runs, count):
    print(f"\n{problem.title}")
    print("update 是减去量；新参数 = 旧参数 - update。空白量用 - 表示。")
    for result in runs:
        print(f"\n{result['optimizer']}  lr={result['lr']:g}")
        print("t param before gradient momentum m v m_hat v_hat scale update after loss_after")
        for row in result["rows"]:
            if row["step"] <= count:
                keys = ("step", "parameter", "parameter_before", "gradient", "momentum_buffer",
                        "m", "v", "m_hat", "v_hat", "direction_scale", "update",
                        "parameter_after", "loss_after")
                print(" ".join(number(row[k]) for k in keys))
        print("最终损失:", number(result["final_loss"]), "参数:",
              ", ".join(number(x) for x in result["parameters"]), "状态:", result["status"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["all"] + list(PROBLEMS), default="all")
    parser.add_argument("--steps", type=int, help="覆盖默认更新次数（不是 epoch）")
    parser.add_argument("--lr", type=float, help="三种优化器使用相同的这个学习率")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-steps", type=int, default=5, help="终端展示前几步；CSV 总是包含全部步骤")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps 必须大于 0")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("--lr 必须为有限正数")
    if not all(0 <= b < 1 for b in (args.beta1, args.beta2, args.momentum)):
        parser.error("beta1/beta2/momentum 必须在 [0, 1) 内")
    if args.print_steps < 0:
        parser.error("--print-steps 不能小于 0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(PROBLEMS.values()) if args.problem == "all" else [PROBLEMS[args.problem]]
    report = ["# 实际运行结果", "", "由 run_experiment.py 生成。终端数值取近似；CSV 保留 Python float 精度。", "",
              f"随机种子：{args.seed}；Adam betas=({args.beta1}, {args.beta2})，eps=1e-8；Momentum mu={args.momentum}。", "",
              "step 表示更新次数；linear 每 3 步为一个 epoch。scalar/anisotropic 为确定性完整目标梯度，只有 linear 使用随机打乱的小批量。", "",
              "三种优化器各自从相同参数及全零状态开始，重用相同批次顺序；梯度随各自的当前位置变化。", ""]
    sweep_rows, config = [], dict(seed=args.seed, beta1=args.beta1, beta2=args.beta2,
                                 momentum=args.momentum, eps=1e-8, data=DATA, problems=[])
    for problem in selected:
        steps = args.steps if args.steps is not None else problem.default_steps
        lr = args.lr if args.lr is not None else problem.default_lr
        schedule = batch_schedule(problem, steps, args.seed)
        config["problems"].append(dict(name=problem.name, start=problem.start, steps=steps, lr=lr,
                                       batches=schedule, sweep_lrs=GRID))
        runs = [run(problem, method, lr, schedule, args.beta1, args.beta2, args.momentum) for method in METHODS]
        print_trace(problem, runs, args.print_steps)
        rows = [row for result in runs for row in result["rows"]]
        fields = list(rows[0]) if rows else ["problem", "optimizer", "step"]
        write_csv(args.out_dir / f"{problem.name}_trace.csv", rows, fields)
        write_loss_svg(args.out_dir / f"{problem.name}_loss.svg", problem, runs)
        report += [f"## {problem.title}", "", f"起点：`{problem.start}`；lr={lr:g}；更新 {steps} 次。", "",
                   "| 优化器 | 初始损失 | 最终损失 | 最终参数 | 完成步数 | 状态 |",
                   "|---|---:|---:|---|---:|---|"]
        for result in runs:
            parameters = ", ".join(f"{n}={number(v)}" for n, v in zip(problem.names, result["parameters"]))
            report.append(f"| {result['optimizer']} | {number(result['initial_loss'])} | {number(result['final_loss'])} | {parameters} | {result['completed_steps']} | {result['status']} |")
        report += ["", f"![完整目标损失曲线]({problem.name}_loss.svg)", "",
                   "### 同一学习率网格，每种优化器分别重置并尝试", "",
                   "每项使用相同步数和批次。以下选择的是该网格上**最终训练损失**最小的一项，不是通用最佳学习率。", "",
                   "| 优化器 | 网格中最佳 lr | 最终训练损失 |", "|---|---:|---:|"]
        for method in METHODS:
            trials = [run(problem, method, rate, schedule, args.beta1, args.beta2, args.momentum) for rate in GRID]
            best = min(trials, key=lambda r: r["final_loss"])
            for trial in trials:
                sweep_rows.append({k: trial[k] for k in ("problem", "optimizer", "lr", "status", "requested_steps", "completed_steps", "initial_loss", "final_loss", "best_loss")})
            if best["status"] == "completed":
                report.append(f"| {method} | {number(best['lr'])} | {number(best['final_loss'])} |")
            else:
                report.append(f"| {method} | 无有限结果 | 所有网格运行均中止 |")
        report.append("")
    write_csv(args.out_dir / "learning_rate_sweep.csv", sweep_rows, list(sweep_rows[0]))
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report += ["## 如何解释", "",
               "- 相同数值 lr 仅便于核对公式；SGD 乘当前梯度，Momentum 乘动量，Adam 乘经二阶矩归一化的一阶矩，实际步长不同。",
               "- 同一批次不等于同一梯度：参数走到不同位置后，各自重新计算梯度。",
               "- 曲线始终比较完整目标损失；linear 的批次损失另列在 CSV，不能混作统一比较指标。",
               "- best_loss 包含 t=0，仅表示运行中最好值；网格选择始终按 final_loss。nonfinite 表示浮点溢出/非有限值并中止，不是严格的数学发散证明。",
               "- 有限步数、选定起点、这一组学习率和无噪声小数据，不足以推出 Adam 一定优于 SGD，也不代表泛化能力、训练耗时或神经网络表现。",
               "- 网格直接看训练目标，没有独立验证/测试集；本实验用于理解更新机制，不是模型选型基准。", ""]
    (args.out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n全部结果已写入：{args.out_dir.resolve()}\n先看 summary.md，再看 *_trace.csv。")


if __name__ == "__main__":
    main()
