"""Module implementing a simple hillclimbing / Simulated Annealing."""

import logging
import numpy as np


logger = logging.getLogger('hillclimb')


def hill_climbing(function, goal, guess,
                  guess_deviation=0.01, goal_delta=0.01, temp_max=5,
                  comparison_method=None,
                  max_iter=100000, rng=None, verbose=False,
                  guess_min=None, guess_max=None):
    """Hill climb to find which is the best value x so that function(x) = goal.

    It is a Simulated Annealing algorithm to avoid getting stuck in local max

    Returns best guess and best result
    """
    if np.any(guess_deviation < 0):
        raise ValueError("The guess deviation should be a "
                         "non zero positive number")
    if goal_delta < 0:
        raise ValueError("The goal error margin (goal_delta) "
                         "should be a positive number")
    if max_iter < 0:
        raise ValueError("The max number of iteration without progress should "
                         "be at least 0")

    if comparison_method is None:
        comparison_method = lambda g, c: np.linalg.norm(g - c, 2)  # noqa
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
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    elif rng is None:
        rng = np.random.RandomState()

    best_guess = np.clip(guess, guess_min, guess_max)
    if not np.allclose(best_guess, guess):
        logger.warning('guess has been clipped.')
    best_res = function(best_guess)
    best_score = comparison_method(goal, best_res)
    init_score = best_score
    i = 0
    while comparison_method(goal, best_res) > goal_delta and i < max_iter:
        cur_guess = rng.multivariate_normal(best_guess, guess_deviation)
        np.clip(cur_guess, guess_min, guess_max, out=cur_guess)
        cur_res = function(cur_guess)
        cur_score = comparison_method(goal, cur_res)
        if temp_max is not None:
            temp = temp_max - temp_max*(i/max_iter)
            simann_bound = np.exp(-(cur_score - best_score)/temp)
        if cur_score < best_score \
                or (temp_max is not None
                    and rng.uniform() < simann_bound):
            best_score = cur_score
            best_res = cur_res
            best_guess = cur_guess
            if verbose and i % 100 == 0:
                print(best_score, '(', i, ')')
        i += 1
        assert init_score >= best_score
    return best_guess, best_res, best_score


def stochastic_hill_climbing(function, goal, guess, guess_deviation=0.01,
                             goal_delta=0.01, comparison_method=None,
                             max_iter=100000, rng=None, verbose=False,
                             guess_min=None, guess_max=None):
    """
    iterative stochastic local search
    selects a random neighbor, compare its distance from the goal,
    if it is better replace the current solution with the neigbhor
    do this multiple times and return the best guess and best score
    """
    
    if np.any(guess_deviation < 0):
        raise ValueError("The guess deviation should be a "
                         "non zero positive number")
    if goal_delta < 0:
        raise ValueError("The goal error margin (goal_delta) "
                         "should be a positive number")
    if max_iter < 0:
        raise ValueError("The max number of iteration without progress should "
                         "be at least 0")

    if comparison_method is None:
        comparison_method = lambda g, c: np.linalg.norm(g - c, 2)  # noqa
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
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    elif rng is None:
        rng = np.random.RandomState()
        
    best_guess = np.clip(guess, guess_min, guess_max)
    if not np.allclose(best_guess, guess):
        logger.warning('guess has been clipped.')
    best_res = function(best_guess)
    best_score = comparison_method(goal, best_res)
    init_score = best_score
    i = 0
    while comparison_method(goal, best_res) > goal_delta and i < max_iter:
        # guess_deviation is a covariance matrix. Its diagonal has the variance values
        cur_guess = rng.multivariate_normal(best_guess, guess_deviation)
        np.clip(cur_guess, guess_min, guess_max, out=cur_guess)
        cur_res = function(cur_guess)
        cur_score = comparison_method(goal, cur_res)
        if cur_score < best_score:
            best_score = cur_score
            best_res = cur_res
            best_guess = cur_guess
            if verbose and i % 100 == 0:
                print(best_score, '(', i, ')')
        i += 1
        assert init_score >= best_score
    return best_guess, best_res, best_score
        
    