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
    fnames = ['fm', 'am', 'entropy', 'goodness', 'amplitude', 'pitch', 'rms']
    if coefs is None:
        coefs = {'fm': 1, 'am': 1, 'entropy': 1, 'goodness': 1,
                 'amplitude': 1, 'pitch': 1, 'rms': 1}
    s_feat = bsa.all_song_features(sig, sr, 
                                   freq_range=bsa.FREQ_RANGE, 
                                   fft_step=bsa.FFT_STEP,
                                   fft_size=bsa.FFT_SIZE)
    if tutor_feat is None:
        features = bsa.normalize_features(s_feat)
    else:
        features = bsa.rescaling_with_tutor_values(tutor_feat, s_feat)
    for key in fnames:
        coef = coefs[key]
        feat = features[key]
        out.append(coef * feat)
    out = np.array(out).T
    return out


def mfcc_measure(sig, sr):
    """Measure the song or song part with mfcc."""
    out = mfcc(sig, sr, numcep=8, appendEnergy=True, winstep=bsa.FFT_STEP/sr,
               winlen=bsa.FFT_SIZE/sr)
    out[:, 0] = bsa.song_amplitude(
        sig, fft_step=bsa.FFT_STEP, fft_size=bsa.FFT_SIZE)[:out.shape[0]]
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
    """Count the number of neighbors of each songs.

    A neighbour is a song of whom the gesture beginnning are close enough
    to those of the song.
    """
    neighbours = np.zeros(len(songs))
    for iref, refsong in enumerate(songs):
        nb_close = 0
        own = [gesture[0] for gesture in refsong.gestures]
        for isong, othersong in enumerate(all_songs):
            if othersong is refsong:
                continue
            song_dist = 0
            other = [gesture[0] for gesture in othersong.gestures]
            for i, start in enumerate(own):
                near_i = bisect_left(other, start)
                if near_i == 0:
                    song_dist += np.abs(start - other[0])
                elif near_i == len(other):
                    song_dist += np.abs(start - other[-1])
                else:
                    song_dist += np.min((np.abs(start - other[near_i - 1]),
                                        np.abs(start - other[near_i])))
            if song_dist < threshold:
                nb_close += 1
        neighbours[iref] = nb_close
    return neighbours


def normalize_and_center(song):
    """Normalize the song between -1 and 1 then centres it around its mean.
    Initialy implemented to normalize the tutor song
    """
    # Normalization
    song = np.array(song, dtype=np.double) # to avoid overflowing calculation
    min_v = song.min()
    max_v = song.max()
    song = 2 * (song - min_v) / (max_v - min_v) - 1
    # Centered with the mean
    song = song - song.mean()
    return song
