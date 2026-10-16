"""数值正确性检查：手算值、解析解、有限差分与批次一致性。"""

import math
import unittest

from optimizers import Adam, Momentum, SGD
from run_experiment import DATA, METHODS, PROBLEMS, batch_schedule, run


class OptimizerTests(unittest.TestCase):
    def test_sgd_first_two_steps_and_analytic_solution(self):
        problem = PROBLEMS["scalar"]
        result = run(problem, "SGD", 0.1, batch_schedule(problem, 20, 7))
        self.assertAlmostEqual(result["rows"][0]["parameter_after"], 0.3)
        self.assertAlmostEqual(result["rows"][1]["parameter_after"], 0.57)
        # 此目标下可以直接解出 w_t = 3*(1-0.9^t)，不是重写训练循环。
        self.assertAlmostEqual(result["parameters"][0], 3 * (1 - 0.9 ** 20), places=13)
        self.assertAlmostEqual(result["final_loss"], 4.5 * 0.9 ** 40, places=13)

    def test_momentum_hand_calculation(self):
        optimizer = Momentum(0.1)
        first = optimizer.step([0.0], [-3.0])[0]
        second = optimizer.step([first.parameter_after], [-2.7])[0]
        self.assertAlmostEqual(first.momentum_buffer, -3)
        self.assertAlmostEqual(second.momentum_buffer, -5.4)
        self.assertAlmostEqual(second.parameter_after, 0.84)

    def test_adam_first_step_and_bias_correction(self):
        first = Adam(0.1).step([0.0], [-3.0])[0]
        self.assertAlmostEqual(first.m, -0.3)
        self.assertAlmostEqual(first.v, 0.009)
        self.assertAlmostEqual(first.m_hat, -3)
        self.assertAlmostEqual(first.v_hat, 9)
        self.assertAlmostEqual(first.parameter_after, 0.09999999966666667, places=14)

    def test_adam_second_step_against_decimal_reference(self):
        # 参考值使用 45 位 Decimal 精度独立计算，固定外部梯度为 2、4。
        optimizer = Adam(0.1)
        first = optimizer.step([0.0], [2.0])[0]
        second = optimizer.step([first.parameter_after], [4.0])[0]
        self.assertAlmostEqual(second.m, 0.58, places=14)
        self.assertAlmostEqual(second.v, 0.019996, places=14)
        self.assertAlmostEqual(second.m_hat, 3.0526315789473684, places=13)
        self.assertAlmostEqual(second.v_hat, 10.003001500750375, places=11)
        self.assertAlmostEqual(second.update, 0.09651820227700591, places=13)

    def test_adam_vector_time_and_per_parameter_state(self):
        optimizer = Adam(0.01)
        updates = optimizer.step([3.0, 3.0], [3.0, 300.0])
        self.assertEqual(optimizer.t, 1)  # 不是“每个参数加 1”
        self.assertAlmostEqual(updates[0].v_hat, 9)
        self.assertAlmostEqual(updates[1].v_hat, 90000)
        self.assertAlmostEqual(updates[0].update, 0.01, places=9)
        self.assertAlmostEqual(updates[1].update, 0.01, places=9)

    def test_momentum_can_continue_when_current_gradient_is_zero(self):
        for optimizer in (Momentum(0.1), Adam(0.1)):
            first = optimizer.step([0.0], [2.0])[0]
            second = optimizer.step([first.parameter_after], [0.0])[0]
            self.assertEqual(second.gradient, 0)
            self.assertGreater(second.update, 0)  # 历史状态仍能推动参数
        self.assertEqual(SGD(0.1).step([1.0], [0.0])[0].update, 0)


class ExperimentTests(unittest.TestCase):
    def test_analytic_gradients_match_finite_differences(self):
        h = 1e-5
        for problem in PROBLEMS.values():
            parameters = [0.4] * len(problem.names)
            batches = [()]
            if problem.name == "linear":
                batches += [(0, 3), (1, 2)]
            for batch in batches:
                gradients = problem.gradient(parameters, batch)
                for i, gradient in enumerate(gradients):
                    plus, minus = parameters.copy(), parameters.copy()
                    plus[i] += h
                    minus[i] -= h
                    finite_difference = (problem.loss(plus, batch) - problem.loss(minus, batch)) / (2 * h)
                    self.assertAlmostEqual(gradient, finite_difference, places=7)

    def test_same_data_batches_and_independent_reproducible_runs(self):
        problem = PROBLEMS["linear"]
        schedule = batch_schedule(problem, 60, 7)
        self.assertEqual(schedule, batch_schedule(problem, 60, 7))
        for start in range(0, 60, 3):
            ids = [i for batch in schedule[start:start + 3] for i in batch]
            self.assertEqual(sorted(ids), list(range(len(DATA))))
        results = [run(problem, method, 0.1, schedule) for method in METHODS]
        batches = [[row["batch_ids"] for row in result["rows"] if row["parameter"] == "w"] for result in results]
        self.assertEqual(batches[0], batches[1])
        self.assertEqual(batches[1], batches[2])
        for method, result in zip(METHODS, results):
            self.assertEqual(result, run(problem, method, 0.1, schedule))
            self.assertAlmostEqual(result["initial_loss"], 59 / 6)
            self.assertAlmostEqual(result["final_loss"], problem.loss(result["parameters"]))

    def test_default_scalar_is_counterexample_to_adam_always_winning(self):
        problem = PROBLEMS["scalar"]
        schedule = batch_schedule(problem, 20, 7)
        sgd, adam = [run(problem, method, 0.1, schedule) for method in ("SGD", "Adam")]
        self.assertLess(sgd["final_loss"], adam["final_loss"])

    def test_numerical_overflow_is_not_reported_as_success(self):
        problem = PROBLEMS["anisotropic"]
        result = run(problem, "SGD", 1.0, batch_schedule(problem, 500, 7))
        self.assertEqual(result["status"], "nonfinite")
        self.assertLess(result["completed_steps"], result["requested_steps"])
        self.assertTrue(math.isinf(result["final_loss"]))
        self.assertTrue(all(math.isfinite(loss) for loss in result["losses"]))


if __name__ == "__main__":
    unittest.main()
