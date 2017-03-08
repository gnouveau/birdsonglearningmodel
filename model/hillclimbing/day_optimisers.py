"""Collection of functions that are day song optimisers."""

import logging
from copy import deepcopy

import numpy as np

from gesture_fitter import fit_gesture_hill

logger = logging.getLogger('DayOptim')
rng = np.random.RandomState()


def optimise_gesture_dummy(songs, tutor_song, measure, comp, train_per_day=10,
                           nb_iter_per_train=10):
    """Optimise gestures randomly from the song models with dummy algorithm."""
    for itrain in range(train_per_day):
        isong = rng.randint(len(songs))
        song = songs[isong]
        ig = rng.randint(len(song.gestures))
        start = song.gestures[ig][0]
        try:
            end = song.gestures[ig + 1][0]
        except IndexError:  # We have picked the last gesture
            end = len(tutor_song)
        logger.info('{}/{}: fit gesture {} of song {} (length {})'.format(
            itrain+1, train_per_day, ig, isong, end-start))
        prior = deepcopy(song.gestures[ig][1])
        res, score = fit_gesture_hill(
            tutor_song[start:end].copy(), measure, comp, start_prior=prior,
            nb_iter=nb_iter_per_train)
        songs[isong].gestures[ig][1] = deepcopy(res)
    return songs


def optimise_gesture_cmaes(songs, tutor_song, measure, comp):
    pass