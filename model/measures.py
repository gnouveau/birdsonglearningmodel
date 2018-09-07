"""Measures for comparing songs and song parts."""

from copy import deepcopy
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


def genetic_neighbours(songs, all_songs, threshold=2000):
    """Count the number of neighbors of each songs.

    A neighbour of a song is a song of whom the gesture's beginnnings
    are close enough to those of the first one.
    """
    neighbours = np.zeros(len(songs))
    for i_ref, ref_song in enumerate(songs):
        nb_close = 0
        own_gest = [gesture[0] for gesture in ref_song.gestures]
        for i_song, other_song in enumerate(all_songs):
            if other_song is ref_song:
                nb_close += 1
                continue
            other_gest = [gesture[0] for gesture in other_song.gestures]
            gest_metric = neighbours_metric(own_gest, other_gest)
            if gest_metric < threshold:
                nb_close += 1
        neighbours[i_ref] = nb_close
    return neighbours


###############################################################################
# SYMMETRICAL DISTANCE FUNCTIONS
###############################################################################

def neighbours_distance(own_gest, other_gest):
    """Measure the symmetrical distance between two songs.
    It is the mean of the metrics between the two songs
    taken in both direction
    """
    metric_own_to_other = neighbours_metric(own_gest, other_gest)
    metric_other_to_own = neighbours_metric(other_gest, own_gest)
    return (metric_own_to_other + metric_other_to_own) / 2

def generate_neighbours_distances_list(ref_song, all_songs):
    """Measure the symmetrical distance bet one song and all the others.
    return a list of distances
    """
    l_neigh_metric = np.zeros(len(all_songs))
    own_gest = [gesture[0] for gesture in ref_song.gestures]
    for i_song, other_song in enumerate(all_songs):
        if other_song is ref_song:
            continue
        other_gest = [gesture[0] for gesture in other_song.gestures]
        gest_dist = neighbours_distance(own_gest, other_gest)
        l_neigh_metric[i_song] = gest_dist
    return l_neigh_metric


def generate_mat_neighbours_distances(songs):
    """Create a matrix with the distances of each song to each others.
    Avoids redundant calculations, because the matrix is symmetrical
    """
    songs = deepcopy(songs)
    mat_neigh_dist = np.zeros((len(songs), len(songs)))
    i_ref = 0
    while len(songs) != 0:
        ref_song = songs[0]
        l_neigh_dist = generate_neighbours_distances_list(ref_song, songs)
        mat_neigh_dist[i_ref, i_ref:] = l_neigh_dist
        mat_neigh_dist[i_ref:, i_ref] = l_neigh_dist
        songs = np.delete(songs, 0)
        i_ref += 1
    return mat_neigh_dist


def update_mat_neighbours_distances(mat_neigh_dist, i_ref, all_songs):
    """Update the line and the column values in the matrix
    relative to one specific song, using the symmetrical distance
    """
    ref_song = all_songs[i_ref]
    mat_neigh_dist = deepcopy(mat_neigh_dist)
    new_list = generate_neighbours_distances_list(ref_song, all_songs)
    mat_neigh_dist[i_ref] = new_list
    mat_neigh_dist[:, i_ref] = new_list
    return mat_neigh_dist

###############################################################################
# ASYMMETRICAL METRIC FUNCTIONS
###############################################################################

def neighbours_metric(own_gest, other_gest):
    """Measure the asymmetrical metric between the gesture's starts of one song
    and the gesture's starts which are the closest among those in the other song
    """
    gest_metric = 0
    for i, start in enumerate(own_gest):
        near_i = bisect_left(other_gest, start)
        if near_i == 0:
            gest_metric += np.abs(start - other_gest[0])
        elif near_i == len(other_gest):
            gest_metric += np.abs(start - other_gest[-1])
        else:
            gest_metric += np.min((np.abs(start - other_gest[near_i - 1]),
                                np.abs(start - other_gest[near_i])))
    return gest_metric


def generate_neighbours_metrics_list(ref_song, all_songs, mirror=False):
    """ Measure the asymmetrical metric between a ref_song and all the others.
    If mirror is True, it will measure for each others songs, their metrics to the ref_song.
    return a list of metrics
    """
    l_neigh_metric = np.zeros(len(all_songs))
    own_gest = [gesture[0] for gesture in ref_song.gestures]
    for i_song, other_song in enumerate(all_songs):
        if other_song is ref_song:
            continue
        other_gest = [gesture[0] for gesture in other_song.gestures]
        if not mirror:
            gest_metric = neighbours_metric(own_gest, other_gest)
        else:
            gest_metric = neighbours_metric(other_gest, own_gest)
        l_neigh_metric[i_song] = gest_metric
    return l_neigh_metric


def generate_mat_neighbours_metrics(songs):
    """"Measure for each song its asymmetrical metric from the others songs.
    return the result as a matrix
    """
    mat_neigh_metric = np.zeros((len(songs), len(songs)))
    for i_ref, ref_song in enumerate(songs):
        l_neigh_metric = generate_neighbours_metrics_list(ref_song, songs)
        mat_neigh_metric[i_ref] = l_neigh_metric
    return mat_neigh_metric


def update_mat_neighbours_metrics(mat_neigh_metric, i_ref, all_songs):
    """Update the line and the column values in the matrix
    relative to one specific song
    """
    ref_song = all_songs[i_ref]
    mat_neigh_metric = deepcopy(mat_neigh_metric)
    line = generate_neighbours_metrics_list(ref_song, all_songs)
    column = generate_neighbours_metrics_list(ref_song, all_songs, mirror=True)
    mat_neigh_metric[i_ref] = line
    mat_neigh_metric[:, i_ref] = column
    return mat_neigh_metric