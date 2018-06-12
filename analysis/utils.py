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
import copy
import json
import pickle
import sys
from scipy.io import wavfile
import birdsonganalysis as bsa

sys.path.append('../model')

from song_model import SongModel
from measures import bsa_measure

sns.set_palette('colorblind')



def _running_mean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
        # TODO: pb en fin de signal avec le debordement d'indice
        y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N


def draw_learning_curve(rd, ax=None):
    score_array = np.array([list(a) for a in rd['scores']]).T
    if ax is None:
        fig = plt.figure(figsize=(16, 5))
        ax = fig.gca()
    for i in range(1, len(rd['scores']), 2):
        ax.axvspan(i, i+1, facecolor='darkblue', alpha=0.1)
    sns.tsplot(score_array, err_style='unit_traces', ax=ax)
    ax.set_xticks(range(0, len(rd['scores']), 20))
    ax.set_xticklabels(range(0, len(rd['scores'])//2, 10))
    ax.set_ylabel('Distance au chant du tuteur')
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



    def show(self, i, vbox):
        try:
            best = np.argmax(self.rd[i]['scores'].iloc[-1])
            vbox.children = [
                self.title(i),
                self.audio(i, -1, best),
                self.tutor_audio(i),
                self.configuration(i),
                self.learning_curve(i),
                self.spec_deriv_plot(i, 0, best), # initial spec deriv
                self.spec_deriv_plot(i, 1, best), # spec deriv of the first day
#                self.spec_deriv_plot(i, 10, best), # ... of the 10th day
                self.spec_deriv_plot(i, -1, best), # ... of the last day
                self.synth_spec(i),
                self.tutor_spec_plot(i),
#                self.gestures_hist(i, -1, best)
            ]
        except NoDataException:
            vbox.children = [
                widgets.HTML('<p>No data yet</p>')
            ]

    def audio(self, irun, iday, ismodel):
        a = Audio(self.rd[irun]['songs'].iloc[iday][ismodel].gen_sound(),
                  rate=44100)
        return widgets.HTML(a._repr_html_())

    def tutor_audio(self, i):
        a = Audio(join(self.run_paths[i], 'tutor.wav'))
        return widgets.HTML(a._repr_html_())

    def spec_deriv_plot(self, irun, iday, ismodel):
        try:
            sm = self.rd[irun]['songs'].iloc[iday][ismodel]
        except IndexError:
            return widgets.HTML('')
        song = sm.gen_sound()
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        ax = bsa.spectral_derivs_plot(bsa.spectral_derivs(song, 256, 40, 1024),
                                      contrast=0.01, ax=ax)
        for start, param in sm.gestures:
            ax.axvline(start//40, color="black", linewidth=1, alpha=0.1)
        #ax.set_title('Spectrogram of model {} on day {} (run {})'.format(
        #    ismodel, iday, irun)
        #)
        ax.set_yticks([])
        ax.set_xticks([])
        if self.save_fig:
            fig.savefig('{}_{}_{}.png'.format(irun, iday, ismodel), dpi=300)
        plt.close(fig)
        return plot_to_html(fig)

    def tutor_spec_plot(self, i):
        sr, tutor = wavfile.read(join(self.run_paths[i], 'tutor.wav'))
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        bsa.spectral_derivs_plot(bsa.spectral_derivs(tutor, 256, 40, 1024),
                                 contrast=0.01, ax=ax)
        gtes = np.loadtxt('../data/{}_gte.dat'.format(
            basename(self.conf[i]['tutor']).split('.')[0]))
        for start in gtes:
            ax.axvline(start//40, color="black", linewidth=1, alpha=0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        if self.save_fig:
            fig.savefig('tutor.png', dpi=300)
        plt.close(fig)
        return plot_to_html(fig)


    def title(self, i):
        return widgets.HTML('<h3>' + self.conf[i]['name'] + '</h3>')

    def configuration(self, i):
        table = widgets.HTML('<table>')
        table.value += '<tr><th>Key</th><th>Value</th></tr>'
        for key in self.conf[i]:
            table.value += '<tr><td>{}</td><td>{}</td>'.format(key, self.conf[i][key])
        table.value += '</table>'
        return table


    def learning_curve(self, i):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        try:
            ax = draw_learning_curve(self.rd[i], ax)
        except Exception as e:
            print(e)
        else:
            sr, synth = wavfile.read('../data/{}_out.wav'.format(
                basename(self.conf[i]['tutor']).split('.')[0]))

            # Do compute score on BA synth only on syllables
            amp = bsa.song_amplitude(synth, 256, 40, 1024)
            sort_amp = np.sort(amp)
            sort_amp = sort_amp[len(sort_amp)//10:]  # discard too low values
            i_max_diff = np.argmax(_running_mean(np.diff(sort_amp), 100))
            threshold = sort_amp[i_max_diff]

            sr, tutor = wavfile.read(join(self.run_paths[i], 'tutor.wav'))
            msynth = bsa_measure(synth, 44100, coefs=self.conf[i]['coefs'])
            mtutor = bsa_measure(tutor, 44100, coefs=self.conf[i]['coefs'])
            score = np.linalg.norm(msynth[amp > threshold] - mtutor[amp > threshold]) / np.sum(amp > threshold) * len(amp)
            ax.axhline(score, color="orange", label="Erreur avec méthode de Boari")
            print("boari score", score)
            ax.legend()
        finally:
            if self.save_fig:
                fig.savefig('learning_curve_{}.png'.format(i), dpi=300)
            plt.close(fig)
        return plot_to_html(fig)        

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
            durations.append((sm.gestures[i+1][0] - sm.gestures[i][0]) /44100 * 1000)
        durations.append((len(sm.song) - sm.gestures[-1][0]) / 44100 * 1000)

        gtes = np.loadtxt('../data/{}_gte.dat'.format(
            basename(self.conf[irun]['tutor']).split('.')[0]))
        tdurations = []
        for i in range(len(gtes) - 1):
            tdurations.append((gtes[i+1] - gtes[i]) / 44100 * 1000)
        tdurations.append((len(sm.song) - gtes[-1]) / 44100 * 1000)
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

    def synth_spec(self, i):
        sr, synth = wavfile.read('../data/{}_out.wav'.format(
            basename(self.conf[i]['tutor']).split('.')[0]))
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()
        bsa.spectral_derivs_plot(bsa.spectral_derivs(synth, 256, 40, 1024),
                                 contrast=0.01, ax=ax)
        gtes = np.loadtxt('../data/{}_gte.dat'.format(
            basename(self.conf[i]['tutor']).split('.')[0]))
        for start in gtes:
            ax.axvline(start//40, color="black", linewidth=1, alpha=0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        if self.save_fig:
            fig.savefig('synth.png', dpi=300)
        plt.close(fig)
        return plot_to_html(fig)
