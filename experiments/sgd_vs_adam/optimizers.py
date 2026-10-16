"""三个可逐行阅读的优化器。只用 Python 标准库，不依赖训练框架。"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence


@dataclass
class Update:
    """一次更新中一个参数的全部中间值。None 在 CSV 中写成空白。"""

    parameter_before: float
    gradient: float
    gradient_sq: float
    direction: float
    direction_scale: float
    update: float  # 有符号的“减去量”；参数的实际变化量是 -update。
    parameter_after: float
    momentum_before: Optional[float] = None
    momentum_buffer: Optional[float] = None
    m_before: Optional[float] = None
    m: Optional[float] = None
    v_before: Optional[float] = None
    v: Optional[float] = None
    m_hat: Optional[float] = None
    v_hat: Optional[float] = None
    denominator: Optional[float] = None


def _check(lr: float, parameters: Sequence[float], gradients: Sequence[float]):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("学习率必须为有限正数")
    if not parameters or len(parameters) != len(gradients):
        raise ValueError("参数与梯度必须非空且长度相同")
    if not all(math.isfinite(x) for x in list(parameters) + list(gradients)):
        raise ValueError("参数与梯度必须是有限数")


class SGD:
    """无动量：theta_new = theta_old - lr * gradient。"""

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> List[Update]:
        _check(self.lr, parameters, gradients)
        rows = []
        for parameter, gradient in zip(parameters, gradients):
            update = self.lr * gradient
            rows.append(Update(parameter, gradient, gradient * gradient,
                               gradient, self.lr, update, parameter - update))
        return rows


class Momentum:
    """SGD + 动量：u = mu*u + g，再减去 lr*u。无 dampening/Nesterov。"""

    def __init__(self, lr: float, mu: float = 0.9):
        if not 0 <= mu < 1:
            raise ValueError("mu 必须在 [0, 1) 内")
        self.lr, self.mu = lr, mu
        self.buffer = []

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> List[Update]:
        _check(self.lr, parameters, gradients)
        if not self.buffer:
            self.buffer = [0.0] * len(parameters)
        if len(self.buffer) != len(parameters):
            raise ValueError("运行中不能改变参数个数")
        rows = []
        for i, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            previous = self.buffer[i]
            self.buffer[i] = self.mu * previous + gradient
            update = self.lr * self.buffer[i]
            rows.append(Update(parameter, gradient, gradient * gradient,
                               self.buffer[i], self.lr, update, parameter - update,
                               momentum_before=previous, momentum_buffer=self.buffer[i]))
        return rows


class Adam:
    """标准 Adam（非 AdamW），m/v 按参数保存；t 每次整组更新只增加一次。"""

    def __init__(self, lr: float, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("beta1/beta2 必须在 [0, 1) 内")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps 必须为有限正数")
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = [], [], 0

    def step(self, parameters: Sequence[float], gradients: Sequence[float]) -> List[Update]:
        _check(self.lr, parameters, gradients)
        if not self.m:
            self.m, self.v = [0.0] * len(parameters), [0.0] * len(parameters)
        if len(self.m) != len(parameters):
            raise ValueError("运行中不能改变参数个数")
        self.t += 1
        rows = []
        for i, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            previous_m, previous_v = self.m[i], self.v[i]
            # 一阶矩：保留有符号的历史梯度，起到平滑方向的作用。
            self.m[i] = self.beta1 * previous_m + (1 - self.beta1) * gradient
            # 二阶原点矩：平方后再平均；不是二阶导数，也不是方差。
            self.v[i] = self.beta2 * previous_v + (1 - self.beta2) * gradient * gradient
            # m/v 从 0 开始，早期偏小；这里进行零初始化偏差修正。
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            denominator = math.sqrt(v_hat) + self.eps  # epsilon 在根号外
            scale = self.lr / denominator
            update = scale * m_hat
            rows.append(Update(parameter, gradient, gradient * gradient,
                               m_hat, scale, update, parameter - update,
                               m_before=previous_m, m=self.m[i],
                               v_before=previous_v, v=self.v[i],
                               m_hat=m_hat, v_hat=v_hat, denominator=denominator))
        return rows
