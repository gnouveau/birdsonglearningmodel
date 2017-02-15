import numpy as np


def hill_climbing(function, goal, guess,
                  guess_deviation=0.01, goal_delta=0.01, temp_max=5,
                  comparison_method=None,
                  max_iter=100000, seed=None, verbose=False,
                  guess_min=None, guess_max=None):
    """
    Do a hill climbing algorithm to find which is the best value x so that
    function(x) = goal

    It is a Simulated Annealing algorithm to avoid getting stuck in local max

    Returns best guess and best result
    """
    if np.any(guess_deviation < 0):
        raise ValueError("The guess deviation should be a non zero positive number")
    if goal_delta < 0:
        raise ValueError("The goal error margin (goal_delta) should be a positive number")
    if max_iter < 1:
        raise ValueError("The max number of iteration without progress should be at least 1")

    if comparison_method is None:
        comparison_method = lambda g, c: np.linalg.norm(g - c, 2)
    if guess_min is None:
        guess_min = -np.inf
    if guess_max is None:
        guess_max = np.inf
    if max_iter is None:
        max_iter = np.inf
    if np.isscalar(guess_deviation):
        guess_deviation = guess_deviation * np.eye(guess.size)
    elif isinstance(guess_deviation, list):
        guess_deviation = np.diag(guess_deviation)
    size = guess.size

    best_res = function(guess)
    best_guess = np.clip(guess, guess_min, guess_max)
    best_score = np.inf


    rng = np.random.RandomState(seed)

    i = 0
    while comparison_method(goal, best_res) > goal_delta and i < max_iter:
        cur_guess = rng.multivariate_normal(best_guess, guess_deviation)
        np.clip(cur_guess, guess_min, guess_max, out=cur_guess)
        cur_res = function(cur_guess)
        cur_score = comparison_method(goal, cur_res)
        temp = temp_max - temp_max*(i/max_iter)
        if (cur_score < best_score
                or rng.uniform() < np.exp(-(cur_score - best_score)/temp)):
            best_score = cur_score
            best_res = cur_res
            best_guess = cur_guess
            if verbose and i % 100 == 0:
                print(best_score, '(', i, ')')
        i += 1
    return best_guess, best_res, best_score