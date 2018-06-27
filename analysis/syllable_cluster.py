"""Analyse syllables."""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import birdsonganalysis as bsa
from copy import deepcopy, copy
from os.path import join, basename
import pandas as pd
import pickle
import sys

sys.path.append('../model')

from song_model import SongModel  # noqa
import birdsonganalysis as bsa


sns.set_palette('colorblind')


def _running_mean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
        y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N


def extract_syllables_feature(song, threshold=None, normalize=False):
    """Extract the features of all syllables and segment them.

    Parameters
    ----------
    song : 1D array_like
        The song to extract syllables from.
    threshold : float, optional
        Threshold for the amplitude of the song to determine when a syllable
        begins and ends. When the amplitude is above the threshold, the
        syllable begins and ends when the amplitude goes under this threshold.
        If threshold is None, then it will be infered by a taking all the
        amplitude sorted and selecting the amplitude for which there is the
        biggest jump in value between this amplitude and the next one.
        The difference are smoothen with a running mean first. See code for
        more details.

    Returns
    -------
    syllables - List of Dict
        Each element of the list is a syllables with the features that have
        been computed for this syllable, in addition to the beg and end of the
        syllable in the measure sampling (one measure every 40 sample of
        sound).

    See also
    --------
    extract_syllables_statistics : compute also the mean and variance of each
        feature.
    all_syllables_features : extract all syllable statistics from a population
        of songs.
    """
    sfeat = bsa.all_song_features(song, bsa.SR, freq_range=bsa.FREQ_RANGE,
                                  fft_size=bsa.FFT_SIZE, fft_step=bsa.FFT_STEP)
    if normalize:
        sfeat = bsa.normalize_features(sfeat)
    if threshold is None:
        sort_amp = np.sort(sfeat['amplitude'])
        sort_amp = sort_amp[len(sort_amp)//10:]  # discard too low values
        i_max_diff = np.argmax(_running_mean(np.diff(sort_amp), 100))
        threshold = sort_amp[i_max_diff]
    syllables = []
    beg = None
    end = None
    for i, amp in enumerate(sfeat['amplitude']):
        if beg is None and amp > threshold:
            beg = i
        elif (beg is not None and
                (amp < threshold or i == len(sfeat['amplitude']) - 1)):
            end = i
            if end - beg > 15:
                syllable_dict = {'beg': beg, 'end': end}
                for key in sfeat:
                    syllable_dict[key] = deepcopy(sfeat[key][beg:end])
                syllables.append(syllable_dict)
            beg = None
            end = None
    return syllables


def percentage_change(first, last, objective=None):
    """Compute the percentage of change between two datasets.

    If objective is set, then a signed value is given for the percentage
    change.
    """
    if objective is not None:
        sign = (np.sign(last.median() - first.median())
                * np.sign(objective.median() - first.median()))
    else:
        sign = np.ones(len(last.median()))
    return sign * ((last.median() - first.median())
                   / first.median() * 100).abs()


def all_syllables_features(rd: pd.DataFrame, progress=None):
    """Get all the information from each syllables from all the songs."""
    syllables = []
    tot = len(rd) * len(rd.iloc[1]['songs'])
    done = 0
    for i, row in rd.iterrows():
        moment = row['moment']
        if moment == 'Start' or moment == 'AfterNight':
            moment = 'morning'
        elif moment == 'End' or moment == 'BeforeNight':
            moment = 'evening'
        for isong, sm in enumerate(row['songs']):
            syllables.extend(extract_syllables_statistics(sm.gen_sound(),
                                                          isong, moment, i))
            done += 1
            if progress is not None:
                progress.value = done / tot
    return pd.DataFrame(syllables)


def extract_syllables_statistics(song, isong=None, moment=None, day=0):
    """Extract syllables from a song and compute the associated statistics."""
    syllables = []
    out = extract_syllables_feature(song)
    for isyb, syllable in enumerate(out):
        if moment == 'evening':
            comb = day//2 + 0.5
        else:
            comb = day//2
        syb_dict = {'day': day//2,
                    'isyb': isyb,
                    'isong': isong,
                    'moment': moment,
                    'comb': comb,
                    'beg': syllable['beg'],
                    'end': syllable['end'],
                    'length': syllable['end'] - syllable['beg']}
        for key in ['fm', 'am', 'entropy', 'goodness', 'amplitude', 'pitch']:
            syb_dict['m'+key] = np.mean(syllable[key])
            syb_dict['v'+key] = np.var(syllable[key])
        syllables.append(syb_dict)
    return syllables


def syllables_from_run(run_path: str, force: bool=False,
                       progress: 'FloatProgress' = None):
    """Load all the syllables from a run and cache them.

    It will extract day by day all the syllables from all the song models of
    the run and return it as a `pandas.DataFrame`. It will also store the
    extraction in a file in the run folder because extraction is very long.

    Parameters
    ----------
    run_path : str
        The location of the directory of the run.
    force : bool
        Force the recomputation of all the syllables even if a cache exists.
    progress : ipywidgets.widgets.FloatProgress
        The progress bar to see the evolution of the extraction. Only useful
        if you run `syllables_from_run` in a Jupyter notebook.

    Returns
    -------
    syllables - pandas.DataFrame
        All the syllables in a DataFrame with their statistics, the day and the
        song model they are from.
    """
    if not force:
        try:
            load = pd.HDFStore(join(run_path, 'syllables.hd5'))
            all_dat = load['syllables']
            if progress is not None:
                progress.value = 1
        except Exception as e:
            print(type(e))
            print(e)
            force = True

    if force:
        try:
            f = open(join(run_path, 'data.pkl'), 'rb')
        except FileNotFoundError:
            f = open(join(run_path, 'data_cur.pkl'), 'rb')
        data = pickle.load(f)
        f.close()
        root_data = [item[1] for item in data if item[0] == 'root']
        rd = pd.DataFrame(root_data)
        all_dat = all_syllables_features(rd, progress)
        all_dat['run_name'] = basename(run_path)
        all_dat.to_hdf(join(run_path, 'syllables.hd5'), 'syllables')
    return all_dat
