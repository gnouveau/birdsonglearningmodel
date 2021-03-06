import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, basename
import pandas as pd
from collections import defaultdict
import io
import base64
from IPython.display import Audio
from ipywidgets import widgets
import json
import pickle
import sys
from scipy.io import wavfile
import birdsonganalysis as bsa

sys.path.append('../model')

from measures import bsa_measure, normalize_and_center

sns.set_palette('colorblind')



def _running_mean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
        # TODO: pb en fin de signal avec le debordement d'indice
        y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N


def boari_synth_song_error(tutor_song, synth_song, p_coefs, tutor_feat):
    """Compute the error distance only on syllables.
    if tutor_feat=None: MAD normalization
    if tutor_feat is defined: rescaling of the features values
    """
    t_amp = tutor_feat["amplitude"]
    amp, th = carac_to_calculate_err_of_synth(synth_song, t_amp)
    msynth = bsa_measure(synth_song, bsa.SR, coefs=p_coefs,
                         tutor_feat=tutor_feat)
    mtutor = bsa_measure(tutor_song, bsa.SR, coefs=p_coefs,
                         tutor_feat=tutor_feat)
    # Calculate the error difference only on the "sound" part
    distance = np.linalg.norm(msynth[amp > th] - mtutor[amp > th])
    # Average the error over all the signal ==> mean error for each sample
    mean_score = distance / np.sum(amp > th) * len(amp)
    return mean_score


def carac_to_calculate_err_of_synth(synth_song, t_amp):
    """
    calculation to do after carac_to_calculate_err_of_synth:
    err_per_feat_synth = err_per_feat(mtutor[amp > th], msynth[amp > th])
    score = np.linalg.norm(msynth[amp > th] - mtutor[amp > th]) / np.sum(amp > th) * len(amp)
    """
    amp = bsa.song_amplitude(synth_song,
                             bsa.FREQ_RANGE,
                             bsa.FFT_STEP,
                             bsa.FFT_SIZE)
    amp = bsa.rescaling_one_feature(t_amp, amp)
    sort_amp = np.sort(amp)
    sort_amp = sort_amp[len(sort_amp)//10:]
    i_max_diff = np.argmax(_running_mean(np.diff(sort_amp), 100))
    th = sort_amp[i_max_diff]
    return amp, th


def err_per_feat(mtutor, msong):
    """
    mtutor and msong are results from bsa_measure().
    """
    err_feats = np.zeros(mtutor.shape[1])
    for i in range(mtutor.shape[1]):
        err_feats[i] = np.sum(np.absolute(mtutor[:,i] - msong[:,i])**2)
    return err_feats


def draw_learning_curve(rd, ax=None):
    # multiply by -1 to have an ascending curve
    # multiple plots
#    score_array = -1 * np.array([list(a) for a in rd['scores']]).T
    # single plot (min)
#    score_array = -1 * np.array([np.amin(scores) for scores in rd['scores']])
    # single plot (mean)
    score_array = -1 * np.array([np.mean(scores) for scores in rd['scores']])
    if ax is None:
        fig = plt.figure(figsize=(16, 5))
        ax = fig.gca()
    for i in range(1, len(rd['scores']), 2):
        ax.axvspan(i, i+1, facecolor='darkblue', alpha=0.1)
    # multiple plots
#    for scores in score_array:
#        plt.plot(scores)
    # single plot
    plt.plot(score_array, label="Chant appris", color='C1')
    ax.set_xticks(range(0, len(rd['scores']), 10))
    ax.set_xticklabels(range(0, len(rd['scores'])//2, 5))
    ax.set_xlim(0, len(rd['scores']))
    ax.set_ylim(-20, -5)
    ax.set_ylabel('Opposé de la distance d\'erreur')
    ax.set_xlabel('Jour')
    ax.set_title('Courbe d\'apprentissage')
    return ax


def plot_to_html(fig):
    # write image data to a string buffer and get the PNG image bytes
    buf = io.BytesIO()
    #fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return widgets.HTML("""<img src='data:image/png;base64,{}'/>""".format(
        base64.b64encode(buf.getvalue()).decode('ascii')))


class NoDataException(Exception):
    pass


class CacheDict(defaultdict):

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


class GridAnalyser:
    """Analyser for the grid search."""

    def __init__(self, run_paths, figsize=(5, 2), save_fig=False):
        """
        run_paths: list of str. list of the names of all the directories with
        the different simulation results.
        ex. of name: seed0+optimise_gesture_whole+default_coef+500_replay+...
        save_fig: boolean. If True save the plot in the current directory
        """
        self.figsize = figsize
        self.data = CacheDict(lambda i: self._get_data(i))
        self.conf = CacheDict(lambda i: self._get_conf(i))
        self.rd = CacheDict(lambda i: self._get_rd(i))
        self.run_paths = run_paths
        self.save_fig = save_fig
        self.options_list = [] # list of sets
        for path in self.run_paths:
            options = basename(path).split('+')
            for i, option in enumerate(options):
                try:
                    self.options_list[i].add(option)
                except IndexError:
                    self.options_list.append(set([option]))
        self.zoom = bsa.FFT_SIZE / bsa.FREQ_RANGE / 4

    def show(self, i, vbox, rescaling=False):
        """
        rescaling: boolean. If True use the rescaling measure to calculate
        the error score for the Boari synthesized song
        """
        try:
            best = np.argmin(self.rd[i]['scores'].iloc[-1])
            mid_i = len(self.rd[i])//2
            if self.conf[i]['days'] % 2 == 0:
                 mid_i -= 1
            vbox.children = [
                self.title(i),
                self.tutor_audio(i),
                self.audio(i, -1, best),
                self.configuration(i),
                self.learning_curve(i, rescaling),  # calculate error score for Boari song
#                self.spec_deriv_plot(i, 0, best),  # initial spec deriv
#                self.spec_deriv_plot(i, mid_i, best),  # spec deriv at the middle of the simulation
                self.spec_deriv_plot(i, -1, best),  # spec deriv at the last day
                self.tutor_spec_plot(i),
                self.synth_spec(i),
                self.song_model_sound_wave(i, -1, best),
                self.tutor_sound_wave(i)
#                self.gestures_hist(i, -1, best)
            ]
        except NoDataException:
            vbox.children = [
                widgets.HTML('<p>No data yet</p>')
            ]
        # save the learned song in a .wav file
#        self.save_audio(i, -1, best)

    def audio(self, irun, iday, ismodel):
        a = Audio(self.rd[irun]['songs'].iloc[iday][ismodel].gen_sound(),
                  rate=bsa.SR)
        return widgets.HTML(a._repr_html_())

    def tutor_audio(self, i):
        a = Audio(join(self.run_paths[i], 'tutor.wav'))
        return widgets.HTML(a._repr_html_())

    def save_audio(self, irun, iday, ismodel):
        sm = self.rd[irun]['songs'].iloc[iday][ismodel]
        soundwave = sm.gen_sound()
        # normalize to get the sound louder
        wav_song = normalize_and_center(soundwave)
        wavfile.write(filename="{}_song.wav".format(self.conf[irun]['name']),
                      rate=bsa.SR,data=wav_song)

    def song_model_sound_wave(self, irun, iday, ismodel):
        sm = self.rd[irun]['songs'].iloc[iday][ismodel]
        soundwave = sm.gen_sound()
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        ax.plot(soundwave, color='C1')
        ax.set_xlim(0, len(soundwave))
        ax.set_title("song model sound wave")
        plt.close(fig)
        return plot_to_html(fig)

    def tutor_sound_wave(self, irun):
        sr, tutor = wavfile.read(join(self.run_paths[irun], 'tutor.wav'))
        tutor = normalize_and_center(tutor)
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        ax.plot(tutor, color='C0')
        ax.set_xlim(0, len(tutor))
        ax.set_title("tutor sound wave (normalized)")
        plt.close(fig)
        return plot_to_html(fig)

    def spec_deriv_plot(self, irun, iday, ismodel):
        try:
            sm = self.rd[irun]['songs'].iloc[iday][ismodel]
        except IndexError:
            return widgets.HTML('')
        song = sm.gen_sound()
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        ax = bsa.spectral_derivs_plot(bsa.spectral_derivs(song,
                                                          bsa.FREQ_RANGE,
                                                          bsa.FFT_STEP,
                                                          bsa.FFT_SIZE),
                                      contrast=0.01, ax=ax)
        for start, param in sm.gestures:
            ax.axvline(start//bsa.FFT_STEP,
                       color="black",
                       linewidth=1,
                       alpha=0.1)
        ax.set_title(self.pretty_title(irun, iday))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, bsa.FREQ_RANGE * self.zoom)
        if self.save_fig:
            fig.savefig('{}_{}_{}.png'.format(irun, iday, ismodel), dpi=300)
        plt.close(fig)
        return plot_to_html(fig)

    def tutor_spec_plot(self, i):
        sr, tutor = wavfile.read(join(self.run_paths[i], 'tutor.wav'))
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        bsa.spectral_derivs_plot(bsa.spectral_derivs(tutor,
                                                     bsa.FREQ_RANGE,
                                                     bsa.FFT_STEP,
                                                     bsa.FFT_SIZE),
                                 contrast=0.01, ax=ax)
        gtes = np.loadtxt('../data/{}_gte.dat'.format(
            basename(self.conf[i]['tutor']).split('.')[0]))
        for start in gtes:
            ax.axvline(start//bsa.FFT_STEP,
                       color="black",
                       linewidth=1,
                       alpha=0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, bsa.FREQ_RANGE * self.zoom)
        ax.set_title("Tutor song spectral derivative")
        if self.save_fig:
            fig.savefig('tutor.png', dpi=300)
        plt.close(fig)
        return plot_to_html(fig)

    def synth_spec(self, i):
        sr, synth = wavfile.read('../data/{}_out.wav'.format(
            basename(self.conf[i]['tutor']).split('.')[0]))
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        bsa.spectral_derivs_plot(bsa.spectral_derivs(synth,
                                                     bsa.FREQ_RANGE,
                                                     bsa.FFT_STEP,
                                                     bsa.FFT_SIZE),
                                 contrast=0.01, ax=ax)
        gtes = np.loadtxt('../data/{}_gte.dat'.format(
            basename(self.conf[i]['tutor']).split('.')[0]))
        for start in gtes:
            ax.axvline(start//bsa.FFT_STEP, color="black", linewidth=1, alpha=0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, bsa.FREQ_RANGE * self.zoom)
        ax.set_title("Boari's synth spectral derivative")
        if self.save_fig:
            fig.savefig('synth.png', dpi=300)
        plt.close(fig)
        return plot_to_html(fig)

    def learning_curve(self, i, rescaling=False):
        """
        rescaling: boolean. If True, use the rescaling measure to calculate
        the error score.
        """
        tutor_feat = None
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        try:
            ax = draw_learning_curve(self.rd[i], ax)
        except Exception as e:
            print(e)
        else:
            sr, synth = wavfile.read('../data/{}_out.wav'.format(
                basename(self.conf[i]['tutor']).split('.')[0]))
            sr, tutor = wavfile.read(join(self.run_paths[i], 'tutor.wav'))
            if rescaling:
                tutor = normalize_and_center(tutor)
                tutor_feat = bsa.all_song_features(tutor, bsa.SR,
                                                   freq_range=bsa.FREQ_RANGE,
                                                   fft_step=bsa.FFT_STEP,
                                                   fft_size=bsa.FFT_SIZE)
            score = boari_synth_song_error(tutor,
                                           synth,
                                           self.conf[i]['coefs'],
                                           tutor_feat)
            ax.axhline(-1 * score, color="orange", label="Erreur avec méthode de Boari")
            print("Boari score:", score)
            best = np.argmin(self.rd[i]['scores'].iloc[-1])
            best_score = self.rd[i]['scores'].iloc[-1][best]
            print("Best song model score:", best_score)
            ax.legend()
        finally:
            if self.save_fig:
                fig.savefig('learning_curve_{}.png'.format(i), dpi=300)
            plt.close(fig)
        return plot_to_html(fig)
        
    def title(self, i):
        return widgets.HTML('<p>' + self.conf[i]['name'] + '</p>')

    def configuration(self, i):
        table = widgets.HTML('<table>')
        table.value += '<tr><th>Key</th><th>Value</th></tr>'
        for key in self.conf[i]:
            table.value += '<tr><td>{}</td><td>{}</td>'.format(key, self.conf[i][key])
        table.value += '</table>'
        return table

    def _get_data(self, i):
        try:
            with open(join(self.run_paths[i], 'data.pkl'), 'rb') as f:
                out = pickle.load(f)
        except FileNotFoundError:
            try:
                with open(join(self.run_paths[i], 'data_cur.pkl'), 'rb') as f:
                    out = pickle.load(f)
            except FileNotFoundError:
                raise NoDataException
        return out

    def gestures_hist(self, irun, iday, ismodel):
        """Plot the histogram of the gesture durations"""
        sm = self.rd[irun]['songs'].iloc[iday][ismodel]
        durations = []
        for i in range(len(sm.gestures) - 1):
            durations.append((sm.gestures[i+1][0] - sm.gestures[i][0]) /bsa.SR * 1000)
        durations.append((len(sm.song) - sm.gestures[-1][0]) / bsa.SR * 1000)

        gtes = np.loadtxt('../data/{}_gte.dat'.format(
            basename(self.conf[irun]['tutor']).split('.')[0]))
        tdurations = []
        for i in range(len(gtes) - 1):
            tdurations.append((gtes[i+1] - gtes[i]) / bsa.SR * 1000)
        tdurations.append((len(sm.song) - gtes[-1]) / bsa.SR * 1000)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        ax1 = sns.distplot(durations, ax=ax1, kde=False)
        ax1.set_title("Distribution des durées de gestes identifiés par notre modèle ({} gestes)".format(len(sm.gestures)))
        ax2 = sns.distplot(tdurations, ax=ax2, kde=False)
        ax2.set_title("Distribution des durées de gestes identifiés par la méthode de Boari ({} gestes)".format(len(gtes)))
        ax2.set_xlabel('Durée (ms)')
        plt.close(fig)
        return plot_to_html(fig)

    def _get_rd(self, i):
        root_data = [item[1] for item in self.data[i] if item[0] == 'root']
        return pd.DataFrame(root_data)

    def _get_conf(self, i):
        with open(join(self.run_paths[i], 'conf.json')) as f:
            conf = json.load(f)
        return conf

    def pretty_title(self, irun, iday):
        """return the moment of the simulation relative to the index"""
        if iday < 0:
            iday = len(self.rd[irun]) + iday
        if iday == 0:
            return "Start"
        elif iday == len(self.rd[irun]) - 1:
            return "End"
        else:
            if iday % 2 ==0:
                moment = "Start"
            else:
                moment = "End"
            return "{} of day {}".format(moment, iday//2 + 1 )