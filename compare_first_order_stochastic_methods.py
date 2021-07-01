import copy
import unittest
from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from evaluator import Evaluator


class AlgEnum(Enum):
    STOCHASTIC_GRADIENT = 0
    STOCHASTIC_MINI_BATCH = 1
    SVRG = 2


class FirstOrderOracle:

    def __init__(self, A: np.array, b: np.array):
        self.A = A
        self.b = b

    def oracle(self, x_t) -> np.array:
        pass


class StochasticGradient(FirstOrderOracle):
    def __init__(self, A: np.array, b: np.array):
        FirstOrderOracle.__init__(self=self, A=A, b=b)
        w, _ = np.linalg.eig(np.transpose(A) @ A)
        self.t = 1
        self.alpha = 2 * min(w)

    def full_gradient(self, A, b, y: np.array) -> np.array:
        return np.transpose(self.A) @ self.A @ y - np.transpose(self.A) @ self.b

    def oracle(self, x_t) -> np.array:
        num_rows, _ = self.A.shape
        chosen_index = np.random.randint(low=0, high=num_rows)
        # stochastic_gradient = num_rows * (np.transpose(self.A[chosen_index]) @ self.A[chosen_index] * x_t -
        #                                   np.transpose(self.A[chosen_index]) * self.b[chosen_index])
        # stochastic_gradient = num_rows * self.full_gradient(A=self.A[chosen_index],b=np.array(self.b[chosen_index]), y=x_t)
        stochastic_gradient = self.full_gradient(A=self.A[chosen_index],b=np.array(self.b[chosen_index]), y=x_t)

        step_size = 2 / (self.alpha * (self.t + 1))
        self.t += 1

        return x_t - step_size * stochastic_gradient


class MiniBatchStochasticGradient(FirstOrderOracle):

    def __init__(self, A: np.array, b: np.array):
        FirstOrderOracle.__init__(self=self, A=A, b=b)
        w, _ = np.linalg.eig(np.transpose(A) @ A)
        self.t = 1
        self.alpha = 2 * min(w)

    def oracle(self, x_t) -> np.array:
        raise NotImplemented(f'Don\'t use me')

    def mini_batch_oracle(self, x_t, batch_size) -> np.array:
        num_rows, _ = self.A.shape
        stochastic_gradient = np.zeros_like(x_t)
        for _ in range(batch_size):
            chosen_index = np.random.randint(low=0, high=num_rows)
            stochastic_gradient += (1 / batch_size) * (
                    np.transpose(self.A[chosen_index]) @ self.A[chosen_index] * x_t -
                    np.transpose(self.A[chosen_index]) * self.b[chosen_index])

        step_size = 2 / (self.alpha * (self.t + 1))
        self.t += 1

        return x_t - step_size * stochastic_gradient


class StochasticVarianceReductionGradient(FirstOrderOracle):
    def __init__(self, A: np.array, b: np.array):
        FirstOrderOracle.__init__(self=self, A=A, b=b)

        w, _ = np.linalg.eig(np.transpose(A) @ A)
        self.t = 1
        self.beta = 2 * max(w)
        self.alpha = 2 * min(w)

        self.k = 20 * self.beta * (1 / self.alpha)
        self.eta = 1 / (10 * self.beta)

    def oracle(self, x_t) -> np.array:
        raise NotImplemented(f'')

    def svrg_internal_oracle(self, x_t, y, full_gradient_at_y) -> np.array:
        num_rows, _ = self.A.shape
        chosen_index = np.random.randint(low=0, high=num_rows)
        sampled_grad_at_x_t = num_rows * (np.transpose(self.A[chosen_index]) @ self.A[chosen_index] * x_t -
                                          np.transpose(self.A[chosen_index]) * self.b[chosen_index])
        sampled_grad_at_y = num_rows * (np.transpose(self.A[chosen_index]) @ self.A[chosen_index] * y -
                                        np.transpose(self.A[chosen_index]) * self.b[chosen_index])
        stochastic_gradient = sampled_grad_at_x_t - sampled_grad_at_y + full_gradient_at_y

        step_size = self.eta

        return x_t - step_size * stochastic_gradient

    def full_gradient(self, y: np.array) -> np.array:
        return np.transpose(self.A) @ self.A @ y - np.transpose(self.A) @ self.b


def generate_random_input(strongly_convex: bool) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    n = 20
    m = 40 if strongly_convex else 1

    default_rng = np.random.default_rng()

    x_1 = default_rng.standard_normal(n)
    x_1 = x_1 / np.linalg.norm(x_1)

    opt = default_rng.standard_normal(n)

    R = np.linalg.norm(opt - x_1)

    A = np.random.normal(size=(m, n))# np.random.rand(m, n)

    noise = default_rng.normal(0.0, 0.5, m)

    b = A @ opt + noise

    # We don't have any bound for G here - since the problem is unconstrained and the gradient is unbounded,
    # G is theoretically infinity.
    # To be able to use the subgradient method, we still need some G, however.
    # Taking G such that the step sizes will be smaller than these of gradient descent, we will get that the
    # subgradient method will be a descent method, in both function values and arguments. Therefore, the distance of the
    # solution from opt will not increase.
    # Therefore, we can assume we stay within a ball of radius ||x_1 - opt||_2 throughout the subgradient algorithm, and
    # set G and D accordingly.
    eigenvalues, _ = np.linalg.eig(np.transpose(A) @ A)
    max_eigenvalue = max(eigenvalues)
    G = 2 * max_eigenvalue * max_eigenvalue * R + 2 * max_eigenvalue * np.linalg.norm(b)

    D = 2 * R

    return A, b, G, D, x_1


def plot_outputs():
    pass


def stochastic_gradient_alg(oracle: StochasticGradient,
                            required_effective_number_of_passes: int,
                            num_iterations_per_effective_pass: int,
                            evaluator: Evaluator,
                            x_1: np.array) -> None:
    x_t = x_1
    running_sum_x_t = x_1

    for effective_iter in range(required_effective_number_of_passes):
        prev_T = ((effective_iter) * int(np.ceil(num_iterations_per_effective_pass)))
        if prev_T > 0.0:
            ratio = (prev_T * (prev_T + 1)) / (new_curr_T * (new_curr_T + 1))
            running_sum_x_t *= ratio
        new_curr_T = ((effective_iter + 1) * int(np.ceil(num_iterations_per_effective_pass)))
        for _ in range(int(np.ceil(num_iterations_per_effective_pass))):
            x_t = oracle.oracle(x_t=x_t)
            running_sum_x_t += 2 * oracle.t * x_t / (new_curr_T * (new_curr_T + 1))

        avg_x_t = running_sum_x_t
        # evaluator.evaluate(curr_iter=effective_iter, curr_xt=avg_x_t)
        evaluator.evaluate(curr_iter=effective_iter, curr_xt=x_t)


def mini_batch_alg(oracle: MiniBatchStochasticGradient,
                   required_effective_number_of_passes: int,
                   num_iterations_per_effective_pass: int,
                   batch_size: int,
                   evaluator: Evaluator,
                   x_1: np.array) -> None:
    x_t = x_1

    for effective_iter in range(required_effective_number_of_passes):
        for _ in range(int(np.ceil(num_iterations_per_effective_pass))):
            x_t = oracle.mini_batch_oracle(x_t=x_t, batch_size=batch_size)
        evaluator.evaluate(curr_iter=effective_iter, curr_xt=x_t)


def svrg(oracle: StochasticVarianceReductionGradient,
         required_effective_number_of_passes: int,
         num_iterations_per_effective_pass: int,
         evaluator: Evaluator,
         x_1: np.array) -> None:
    y = x_1
    x_t = y

    for effective_iter in range(required_effective_number_of_passes):
        for _ in range(num_iterations_per_effective_pass):
            gradient_at_y = oracle.full_gradient(y)
            next_y = np.zeros_like(y)
            for _ in range(int(np.ceil(oracle.k))):
                x_t = oracle.svrg_internal_oracle(x_t=x_t, y=y, full_gradient_at_y=gradient_at_y)
                next_y += (1 / np.ceil(oracle.k)) * x_t

            y = next_y

        evaluator.evaluate(curr_iter=effective_iter, curr_xt=y)


def main():
    strongly_convex_simulations_num = 1
    external_iterations_num = 10

    batch_sizes = [2, 5, 10, 20, 50]

    for iteration in range(strongly_convex_simulations_num):
        A, b, G, D, x_1 = generate_random_input(strongly_convex=True)

        optimal_solution_arg = np.linalg.pinv(np.transpose(A) @ A) @ np.transpose(A) @ b
        optimal_solution_val = 0.5 * np.power(np.linalg.norm(A @ optimal_solution_arg - b), 2)

        svrg_oracle = StochasticVarianceReductionGradient(A=A, b=b)
        print(f'optimal_solution_val = {optimal_solution_val}')
        print(f'SVRG\'s k={svrg_oracle.k}')

        iterations_num = 50 * int(np.ceil(svrg_oracle.k)) * external_iterations_num
        evaluator_stochastic_gradient = Evaluator(A=A, b=b, iteration_num=external_iterations_num,
                                                  optimal_solution_val=optimal_solution_val)

        stochastic_gradient_alg(oracle=StochasticGradient(A=A, b=b),
                                required_effective_number_of_passes=external_iterations_num,
                                num_iterations_per_effective_pass=50 * svrg_oracle.k,
                                x_1=copy.deepcopy(x_1),
                                evaluator=evaluator_stochastic_gradient)

        mini_batch_evaluators = []
        for mini_batch_size in batch_sizes:
            mini_batch_evaluator = Evaluator(A=A, b=b, iteration_num=external_iterations_num,
                                             optimal_solution_val=optimal_solution_val)
            mini_batch_evaluators.append(mini_batch_evaluator)
            mini_batch_alg(MiniBatchStochasticGradient(A=A, b=b),
                           required_effective_number_of_passes=external_iterations_num,
                           num_iterations_per_effective_pass=(50 * svrg_oracle.k) / mini_batch_size,
                           batch_size=mini_batch_size,
                           evaluator=mini_batch_evaluator,
                           x_1=copy.deepcopy(x_1))

        svrg_evaluator = Evaluator(A=A, b=b, iteration_num=external_iterations_num,
                                   optimal_solution_val=optimal_solution_val)

        svrg(oracle=svrg_oracle, required_effective_number_of_passes=external_iterations_num,
             num_iterations_per_effective_pass=50, evaluator=svrg_evaluator, x_1=copy.deepcopy(x_1))

        plt.plot(evaluator_stochastic_gradient.scores, label='Stochastic Gradient')
        for i in range(len(batch_sizes)):
            batch_size = batch_sizes[i]
            mini_batch_evaluator = mini_batch_evaluators[i]
            plt.plot(mini_batch_evaluator.scores, label=f'Mini Batch - Batch={batch_size}')
        plt.plot(svrg_evaluator.scores, label='SVRG')
        plt.title(f'##')
        plt.xlabel('iterations')
        plt.ylabel('distance from optimum')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()

