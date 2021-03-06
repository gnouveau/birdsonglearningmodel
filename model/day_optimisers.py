"""Collection of functions that are day song optimisers."""

import logging
from copy import deepcopy

import numpy as np

from datasaver import QuietDataSaver
from gesture_fitter import fit_gesture_hill, fit_gesture_padded, \
                           fit_gesture_whole, fit_gesture_whole_local_search, \
                           _padded_gen_sound
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


def optimise_gesture_padded(songs, tutor_song, conf, datasaver=None, iday=None):
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


def optimise_gesture_whole(songs, goal, conf, datasaver=None, iday=None):
    """Optimise gestures randomly from the song models with dummy algorithm.

    Include the previous and next gesture in the evaluation to remove
    border issues.
    """
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    train_per_day = conf['train_per_day']
    rng = conf['rng_obj']
    tutor_song = songs[0].song # just for the assert
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    for itrain in range(train_per_day):
        isong = rng.randint(len(songs))
        song = songs[isong]
        ig = rng.randint(len(song.gestures))
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


def optimise_gesture_whole_local_search(songs, goal, conf,
                                        datasaver=None, iday=None):
    """
    Optimise gestures randomly from the song models
    with a stochastic local search
    """
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    train_per_day = conf['train_per_day']
    rng = conf['rng_obj']
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    improvement_cpt = np.zeros(train_per_day)
    for itrain in range(train_per_day):
        isong = rng.randint(len(songs))
        song = songs[isong]
        ig = rng.randint(len(song.gestures))
        s = song.gen_sound()
        c = measure(s)
        pre_score = comp(goal, c)
        logger.info('{}/{}: fit gesture {} of song {} (length {}, score {})'.format(
            itrain+1, train_per_day, ig, isong, len(s), pre_score))
        res, hill_score = fit_gesture_whole_local_search(
            goal, song, ig, conf)
        songs[isong].gestures[ig][1] = deepcopy(res)
#        datasaver.add(iday=iday, itrain=itrain, isong=isong, ig=ig,
#                      pre_score=pre_score, new_score=hill_score)  # too much verbose
        logger.info('new score {}'.format(hill_score))
        assert pre_score >= hill_score, "{} >= {} est faux".format(
            pre_score, hill_score)
        if hill_score < pre_score:
            improvement_cpt[itrain] += 1
    datasaver.add(label='day', cond='after_day_learning',
                  improvement_cpt=improvement_cpt)
    return songs


def optimise_root_mean_square_error(songs, tutor_song, conf, 
                                    datasaver=None, iday=None):
    """Optimises the gestures so that each sample of the song model
    is as close as possible to the corresponding sample in the tutor song.
    Only use the root mean square error and not some features of the song.
    """
    comp = conf["comp_obj"]
    train_per_day = conf["train_per_day"]
    nb_iter = conf["iter_per_train"]
    rng = conf["rng_obj"]
    deviation = np.diag(conf["dev"])
    
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()

    for itrain in range(train_per_day):
        isong = rng.randint(len(songs))
        song = songs[isong]
        ig = rng.randint(len(song.gestures))
        s = song.gen_sound()
        pre_score = comp(tutor_song, s)
        logger.info('{}/{}: fit gesture {} of song {} (length {}, score {})'.format(
            itrain+1, train_per_day, ig, isong, len(s), pre_score))
        best_gest = deepcopy(song.gestures[ig][1])
        best_score = pre_score
        i = 0
        while best_score > 0.01 and i < nb_iter:
            new_gest = rng.multivariate_normal(best_gest, deviation)
            new_sound = _padded_gen_sound(song, range(0, len(song.gestures)),
                                         ig, new_gest)
            new_score = comp(tutor_song, new_sound)
            if new_score < best_score:
                best_score = new_score
                best_gest = new_gest
            i += 1
        song.gestures[ig][1] = deepcopy(best_gest)
#        datasaver.add(iday=iday, itrain=itrain, isong=isong, ig=ig,
#                      pre_score=pre_score, new_score=best_score)  # too much verbose
        logger.info("new score {}".format(best_score))
    return songs


def optimise_proportional_training(songs, goal, conf,
                                        datasaver=None, iday=None):
    """
    Optimise gestures randomly from the song models
    with a stochastic local search
    The number of trainings for this day is proportional to
    the total number of gestures in the day songs
    """
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    rng = conf['rng_obj']
    train_multiplier = conf.get('train_multiplier', 3)
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    train_per_day = 0
    for song in songs:
        train_per_day += len(song.gestures)
    train_per_day *= train_multiplier
    train_per_day = round(train_per_day)  # avoid possible float value with train_multiplier < 1
    datasaver.add(label='day', cond='define_number_of_trainings',
                  train_per_day=train_per_day)
    improvement_cpt = np.zeros(train_per_day)
    for itrain in range(train_per_day):
        isong = rng.randint(len(songs))
        song = songs[isong]
        ig = rng.randint(len(song.gestures))
        s = song.gen_sound()
        c = measure(s)
        pre_score = comp(goal, c)
        logger.info('{}/{}: fit gesture {} of song {} (length {}, score {})'.format(
            itrain+1, train_per_day, ig, isong, len(s), pre_score))
        res, hill_score = fit_gesture_whole_local_search(
            goal, song, ig, conf)
        songs[isong].gestures[ig][1] = deepcopy(res)
#        datasaver.add(iday=iday, itrain=itrain, isong=isong, ig=ig,
#                      pre_score=pre_score, new_score=hill_score)  # too much verbose
        logger.info('new score {}'.format(hill_score))
        assert pre_score >= hill_score, "{} >= {} est faux".format(
            pre_score, hill_score)
        if hill_score < pre_score:
            improvement_cpt[itrain] += 1
    datasaver.add(label='day', cond='after_day_learning',
                  improvement_cpt=improvement_cpt)
    return songs


def optimise_gesture_cmaes(songs, tutor_song, measure, comp):
    """Optimise gestures guided with a CMA-ES algorithm."""
    raise NotImplementedError()
