import unittest

from compare_first_order_stochastic_methods import generate_random_input, StochasticGradient, \
    StochasticVarianceReductionGradient, MiniBatchStochasticGradient
import numpy as np

class test_StochasticGradient(unittest.TestCase):

    def test_stochastic_gradient(self):

        A, b, G, D, x_1 = generate_random_input(strongly_convex=True)
        stochastic_gradient_oracle = StochasticGradient(A=A,b=b)
        svrg_oracle = StochasticVarianceReductionGradient(A=A, b=b)
        real_gradient = svrg_oracle.full_gradient(x_1)

        avg_gradient = np.zeros_like(x_1)

        iterations_num = 1000
        for _ in range(iterations_num):
            x_2 = stochastic_gradient_oracle.oracle(x_1)
            avg_gradient += (1 / iterations_num) * (x_1 - x_2) * stochastic_gradient_oracle.alpha
            stochastic_gradient_oracle.t -= 1

        self.assertAlmostEqual(np.linalg.norm(real_gradient - avg_gradient), 0.0)

    def test_mini_batch_gradient(self):

        A, b, G, D, x_1 = generate_random_input(strongly_convex=True)
        stochastic_gradient_oracle = MiniBatchStochasticGradient(A=A,b=b)
        svrg_oracle = StochasticVarianceReductionGradient(A=A, b=b)
        real_gradient = svrg_oracle.full_gradient(x_1)

        avg_gradient = np.zeros_like(x_1)

        iterations_num = 1000
        batch_size = 10
        for _ in range(iterations_num):
            x_2 = stochastic_gradient_oracle.mini_batch_oracle(x_1, batch_size=batch_size)
            avg_gradient += (1 / iterations_num) * (x_1 - x_2) * stochastic_gradient_oracle.alpha
            stochastic_gradient_oracle.t -= 1

        self.assertAlmostEqual(np.linalg.norm(real_gradient - avg_gradient), 0.0)