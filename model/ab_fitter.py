"""Module that fit an equation to alpha and beta raw data."""

import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from synth import f, f_str
from fastdtw import fastdtw

from hill_climbing import hill_climbing
import birdsonganalysis as bsa

sns.set_context('colorblind')


def fit_func(goal, f, params, dev, mins, maxs):
    """
    Fit params of so that f(i) is the closest to goal[i].

    Find the params so that the function f fits the best the goal by
    minimizing square error, return the best params.

    goal - An array of the value that the function must fit.
           Assume f(i) = goal[i]
    f - The function to fit. must be of the signature f(i, params) with params
        being an ndarray.
    params - Either guesses for the params as an ndarray, or the number of
             params to fit.
    returns best_params, sq_error
    """
    try:
        nb_params = len(params)
        guess = copy.copy(params)
    except TypeError:
        nb_params = params
        guess = np.ones(nb_params)
    params_out, y, score = hill_climbing(
        lambda param: f(np.arange(len(goal)), param),
        goal, guess, guess_deviation=dev,
        comparison_method=lambda g, c: fastdtw(g, c, dist=2)[0],
        max_iter=10000,
        guess_min=np.array(mins),
        guess_max=np.array(maxs),
        temp_max=2)
    return params_out, score


if __name__ == "__main__":
    def _alphaf(x, p):
        x = x/bsa.SR
        return f(x, p, 1, 3)

    goal = np.loadtxt('../../data/ba_syllable_a_end_ab.dat')
    i = 1
    j = 3
    prior = []
    dev = []
    mins = []
    maxs = []
    for k in range(1, i + 1):  # prior on exponential
        prior.extend([2/k, 100/k])
        dev.extend([0.05, 0.05])
        mins.extend([-10, 0])
        maxs.extend([10, 5000])
    for k in range(1, j + 1):  # prior on sin
        prior.extend([1/k, 0, 200*k**2])
        dev.extend([0.1/k, 0.005, 5*k])
        mins.extend([0, -np.pi, 0])
        maxs.extend([10, np.pi, 8000])
    prior.append(4)
    mins.append(0)
    maxs.append(10)
    dev.append(0.5)

    bprior = [0, 0, 0, 1, 100, -0.028]  # beta prior
    bdev = [0.01, 0.01, 0.01, 0.01, 10, 0.001]
    bmins = [-10, 0, 0, -np.pi, 0, -3]
    bmaxs = [10, 10, 3, np.pi, 1000, 0]

    pout, score = fit_func(goal[:, 0], _alphaf, np.array(prior), np.diag(dev),
                           mins, maxs)

    plt.plot(goal[:, 0])
    plt.plot(_alphaf(np.arange(goal.shape[0]), pout))
    plt.show()
    print(f_str(pout, 1, 3, 't'))
