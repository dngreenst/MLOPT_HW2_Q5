import numpy as np


class Evaluator:
    def __init__(self, A: np.array, b: np.array, iteration_num, optimal_solution_val):
        self.A = A
        self.b = b
        self.iteration_num = iteration_num
        self.scores = np.zeros(self.iteration_num)
        self.optimal_solution_val = optimal_solution_val

    def evaluate(self, curr_iter, curr_xt):
        score = 0.5 * np.power(np.linalg.norm(self.A @ curr_xt - self.b), 2) - self.optimal_solution_val

        self.scores[curr_iter] = score
