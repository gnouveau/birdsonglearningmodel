"""Measures for comparing songs and song parts."""

import numpy as np
import birdsonganalysis as bsa
from python_speech_features import mfcc
from bisect import bisect_left

def _running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def bsa_measure(sig, sr, coefs=None, tutor_feat=None):
    """Measure the song or song part with standard birdsong analysis."""
    out = []
    fnames = ['fm', 'am', 'entropy', 'goodness', 'amplitude', 'pitch']
    if coefs is None:
        coefs = {'fm': 1, 'am': 1, 'entropy': 1, 'goodness': 1,
                 'amplitude': 1, 'pitch': 1}
    if tutor_feat is None:
        features = bsa.normalize_features(
                bsa.all_song_features(sig, sr, freq_range=256, fft_step=40,
                                      fft_size=1024))
    else:
        features = bsa.rescaling_with_tutor_values(tutor_feat,
                bsa.all_song_features(sig, sr, freq_range=256, fft_step=40,
                                      fft_size=1024))
    for key in fnames:
        coef = coefs[key]
        feat = features[key]
        out.append(coef * feat)
    out = np.array(out).T
    return out


def mfcc_measure(sig, sr):
    """Measure the song or song part with mfcc."""
    out = mfcc(sig, sr, numcep=8, appendEnergy=True, winstep=40/sr,
               winlen=1024/sr)  # FIXME should not be hardwritten 40, 1024
    out[:, 0] = bsa.song_amplitude(
        sig, fft_step=40, fft_size=1024)[:out.shape[0]]
    return out


def get_scores(goal, song_models, measure, comp):
    """Get the score of each model compared to the tutor song.

    tutor_song - The signal of the tutor song
    song_models - A list of all the song models
    """
    scores = np.zeros(len(song_models))
    for i in range(len(song_models)):
        sig = song_models[i].gen_sound()
        c = measure(sig)
        scores[i] = comp(goal, c)
    return scores


def genetic_neighbours(songs, all_songs, threshold=2000):
    neighbours = np.ones(len(songs))
    for iref, refsong in enumerate(songs):
        nb_close = 0
        """
        TODO: du coup il se mesure a lui meme, donc forcement
        dans un tour de boucle, song_dist sera egal a zero et donc nb_close
        faudra au moins 1 ==> est ce problematique ?
        """
        for isong, othersong in enumerate(all_songs):
            song_dist = 0
            own = [gesture[0] for gesture in refsong.gestures]
            for i, gesture in enumerate(othersong.gestures):
                start = gesture[0]
                near_i = bisect_left(own, start)
                if near_i >= len(own) - 1:
                    near_i = len(own) - 2
                cur_dist = np.min((np.abs(start - own[near_i]),
                                   np.abs(start - own[near_i+1])))
                song_dist += cur_dist
            if song_dist < threshold:
                nb_close += 1
        neighbours[iref] = nb_close
    return neighbours
