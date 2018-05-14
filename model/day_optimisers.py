"""Collection of functions that are day song optimisers."""

import logging
from copy import deepcopy

import numpy as np

from datasaver import QuietDataSaver
from gesture_fitter import fit_gesture_hill, fit_gesture_padded, \
                           fit_gesture_whole
from synth import gen_sound, only_sin

logger = logging.getLogger('DayOptim')


def optimise_gesture_dummy(songs, tutor_song, measure, comp, train_per_day=10,
                           nb_iter_per_train=10, datasaver=None, rng=None):
    """Optimise gestures randomly from the song models with dummy algorithm."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
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
        g = measure(tutor_song[start:end])
        s = gen_sound(
            prior, end - start,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13)
        c = measure(s)
        pre_score = comp(g, c)
        res, hill_score = fit_gesture_hill(
            tutor_song[start:end].copy(), measure, comp, start_prior=prior,
            nb_iter=nb_iter_per_train, temp=None, rng=rng)
        # datasaver.add(pre_score=pre_score,
        #               new_score=hill_score, isong=isong, ig=ig)
        songs[isong].gestures[ig][1] = deepcopy(res)
        assert pre_score >= hill_score
    return songs


def optimise_gesture_padded(songs, tutor_song, conf, datasaver=None):
    """Optimise gestures randomly from the song models with dummy algorithm.

    Include the previous and next gesture in the evaluation to remove
    border issues.
    """
    nb_pad = conf.get('nb_pad', 2)
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    train_per_day = conf['train_per_day']
    rng = conf['rng_obj']
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    for itrain in range(train_per_day):
        isong = rng.randint(len(songs))
        song = songs[isong]
        ig = rng.randint(len(song.gestures))
        if ig-nb_pad >= 0: # TODO: a confirmer : le padding permet de prendre un geste plus long pour eviter les effets de bords. Oui a priori
            start = song.gestures[ig - nb_pad][0]
        else:
            start = 0
        end = song.gesture_end(ig + nb_pad) # the potential index out of bound is handled by the gesture_end function
        logger.info('{}/{}: fit gesture {} of song {} (length {})'.format(
            itrain+1, train_per_day, ig, isong, end-start))
        g = measure(tutor_song[start:end])
        range_ = range(max(0, ig-nb_pad), min(len(song.gestures)-1, ig+nb_pad)+1)
        s = song.gen_sound(range_)
        assert len(tutor_song[start:end]) == len(s), "%d %d" % (end - start, len(s))
        c = measure(s)
        pre_score = comp(g, c)
        res, hill_score = fit_gesture_padded(
            tutor_song, song, ig, conf)
        # datasaver.add(pre_score=pre_score,
        #               new_score=hill_score, isong=isong, ig=ig)
        songs[isong].gestures[ig][1] = deepcopy(res)
        assert pre_score >= hill_score, "{} >= {} est faux".format(
            pre_score, hill_score)
    return songs


def optimise_gesture_whole(songs, tutor_song, conf, datasaver=None):
    """Optimise gestures randomly from the song models with dummy algorithm.

    Include the previous and next gesture in the evaluation to remove
    border issues.
    """
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    train_per_day = conf['train_per_day']
    rng = conf['rng_obj']
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    for itrain in range(train_per_day):
        isong = rng.randint(len(songs))
        song = songs[isong]
        ig = rng.randint(len(song.gestures))
        goal = measure(tutor_song)
        s = song.gen_sound()
        assert len(tutor_song) == len(s), "%d %d" % (len(tutor_song), len(s))
        c = measure(s)
        pre_score = comp(goal, c)
        logger.info('{}/{}: fit gesture {} of song {} (length {}, score {})'.format(
            itrain+1, train_per_day, ig, isong, len(s), pre_score))
        res, hill_score = fit_gesture_whole(
            goal, song, ig, conf)
        # datasaver.add(pre_score=pre_score,
        #               new_score=hill_score, isong=isong, ig=ig)
        songs[isong].gestures[ig][1] = deepcopy(res)
        logger.info('new score {}'.format(hill_score))
        assert pre_score >= hill_score, "{} >= {} est faux".format(
            pre_score, hill_score)
    return songs


def optimise_gesture_cmaes(songs, tutor_song, measure, comp):
    """Optimise gestures guided with a CMA-ES algorithm."""
    raise NotImplementedError()
