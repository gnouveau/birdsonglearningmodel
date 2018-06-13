"""Fit a gesture to real sound."""
from copy import deepcopy
import numpy as np

from hill_climbing import hill_climbing, stochastic_hill_climbing
from synth import gen_sound, only_sin, gen_alphabeta, synthesize


def fit_gesture_hill(gesture, conf, prior):
    """Find the parameters to fit to a gesture."""
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_iter = conf['iter_per_train']
    temp = conf.get('temperature', None)
    rng = conf.get('rng', None)
    size = len(gesture)
    goal = measure(gesture)
    x, dummy_y, score = hill_climbing(
        function=lambda x: measure(gen_sound(
            x, size,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13)),
        goal=goal,
        guess=np.array(prior),
        guess_min=conf['mins'],
        guess_max=conf['maxs'],
        guess_deviation=np.diag(conf['dev']),
        max_iter=nb_iter,
        comparison_method=comp,
        temp_max=temp,
        verbose=False,
        rng=rng)
    return x, score


def _padded_gen_sound(songmodel, range_, change_index, param, out_ab=False):
    """
    Modifies one syllable of a song.
    Creates a new song based on `songmodel` and where the `change_index` 
    syllable is replaced by a new one created by `falpha` and `fbeta`
    using `param`
    """
    alpha_betas = []
    for i in range_[:-1]:
        if i != change_index:
            alpha_betas.append(songmodel.gen_alphabeta(range_=[i], pad=False))
        else:
            start = songmodel.gestures[i][0]
            end = songmodel.gesture_end(i)
            size = end - start
            alpha_betas.append(gen_alphabeta(
                param, size,
                falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                falpha_nb_args=13, pad=False, beg=0))
    # last one with padding
    i = range_[-1]
    if i != change_index:
        alpha_betas.append(songmodel.gen_alphabeta(range_=[i], pad=True))
    else:
        start = songmodel.gestures[i][0]
        end = songmodel.gesture_end(i)
        size = end - start
        alpha_betas.append(gen_alphabeta(
            param, size,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13, pad=True, beg=0))
    if out_ab:
        return synthesize(np.concatenate(alpha_betas),
                          fixed_normalize=True), np.concatenate(alpha_betas)
    else:
        return synthesize(np.concatenate(alpha_betas), fixed_normalize=True)


def fit_gesture_padded(tutor, songmodel, gesture_index, conf):
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_iter = conf['iter_per_train']
    nb_pad = conf.get('nb_pad', 1)
    temp = conf.get('temperature', None)
    rng = conf.get('rng', None)

    prev_igest = max(0, gesture_index - nb_pad)
    start_tutor = songmodel.gestures[prev_igest][0]
    next_igest = min(len(songmodel.gestures) - 1, gesture_index + nb_pad)
    end_tutor = songmodel.gesture_end(next_igest)
    goal = measure(tutor[start_tutor:end_tutor])
    x, dummy_y, score = hill_climbing(
        function=lambda x: measure(_padded_gen_sound(
            songmodel,
            range(prev_igest, next_igest+1),
            gesture_index,
            x)),
        goal=goal,
        guess=deepcopy(songmodel.gestures[gesture_index][1]),
        guess_min=conf['mins'],
        guess_max=conf['maxs'],
        guess_deviation=np.diag(conf['dev']),
        max_iter=nb_iter,
        comparison_method=comp,
        temp_max=temp,
        verbose=False,
        rng=rng)
    return x, score


def fit_gesture_whole(measured_tutor, songmodel, gesture_index, conf):
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_iter = conf['iter_per_train']
    temp = conf.get('temperature', None)
    rng = conf.get('rng', None)
    goal = measured_tutor
    x, dummy_y, score = hill_climbing(
        function=lambda x: measure(_padded_gen_sound(
            songmodel,
            range(0, len(songmodel.gestures)),
            gesture_index,
            x)),
        goal=goal,
        guess=deepcopy(songmodel.gestures[gesture_index][1]),
        guess_min=conf['mins'],
        guess_max=conf['maxs'],
        guess_deviation=np.diag(conf['dev']),
        max_iter=nb_iter,
        comparison_method=comp,
        temp_max=temp,
        verbose=False,
        rng=rng)
    return x, score

def fit_gesture_whole_local_search(measured_tutor, songmodel, gesture_index, conf):
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_iter = conf['iter_per_train']
    rng = conf.get('rng', None)
    x, dummy_y, score = stochastic_hill_climbing(
        function=lambda x: measure(_padded_gen_sound(
            songmodel,
            range(0, len(songmodel.gestures)),
            gesture_index,
            x)),
        goal=measured_tutor,
        guess=deepcopy(songmodel.gestures[gesture_index][1]),
        guess_min=conf['mins'],
        guess_max=conf['maxs'],
        guess_deviation=np.diag(conf['dev']),
        max_iter=nb_iter,
        comparison_method=comp,
        verbose=False,
        rng=rng)
    return x, score