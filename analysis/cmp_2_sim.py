import json
import matplotlib.pyplot as plt
import numpy as np
from os.path import basename, join
import pandas
import pickle
from scipy.io import wavfile
import sys
import warnings

import birdsonganalysis as bsa
import utils

sys.path.append('../model/')
from measures import bsa_measure, normalize_and_center

def get_run_param_and_songlog(path):
    with open(join(path, 'conf.json'), 'r') as f:
        run_param = json.load(f)

    try:
        with open(join(path, 'data.pkl'), 'rb') as f:
            songlog = pickle.load(f)
    except FileNotFoundError:
        try:
            warnings.warn('Learning not over')
            with open(join(path, 'data_cur.pkl'), 'rb') as f:
                songlog = pickle.load(f)
        except FileNotFoundError:
            print("Error: no data files")
    
    return run_param, songlog

def get_rd_best_smodel_and_score(songlog):
    root_data = [item[1] for item in songlog if item[0] == 'root']
    rd = pandas.DataFrame(root_data)
    best = np.argmin(rd['scores'].iloc[-1])
    smodel = rd['songs'].iloc[-1][best]
    score = rd['scores'].iloc[-1][best]
    return rd, smodel, score

def get_features(song, param_feat):
    song_feat = bsa.all_song_features(song,
                                      bsa.SR,
                                      freq_range=bsa.FREQ_RANGE,
                                      fft_step=bsa.FFT_STEP,
                                      fft_size=bsa.FFT_SIZE)
    return bsa.rescaling_with_tutor_values(param_feat, song_feat)

def generate_data_struct(l_path):
    sim = []
    for path in l_path:
        d = {}
        d["fft_step"] = bsa.FFT_STEP
        d["freq_range"] = bsa.FREQ_RANGE
        d["fft_size"] = bsa.FFT_SIZE
        d["sr"], d["tutor"] = wavfile.read(join(path, 'tutor.wav'))
        d["tspec"] = bsa.spectral_derivs(d["tutor"],
                                         d["freq_range"],
                                         d["fft_step"],
                                         d["fft_size"])
        d["run_param"], d["songlog"] = get_run_param_and_songlog(path)
        rd, smodel, score = get_rd_best_smodel_and_score(d["songlog"])
        d["rd"] = rd
        d["smodel"] = smodel
        d["score"] = score
        d["song"] = smodel.gen_sound()
        d["starts"] = []
        for i, gesture in enumerate(d["smodel"].gestures):
            d["starts"].append(gesture[0])
        d["smspec"] = bsa.spectral_derivs(d["song"],
                                          d["freq_range"],
                                          d["fft_step"],
                                          d["fft_size"])
        song_name = basename(d["run_param"]['tutor']).split('.')[0]
        synth_ab = np.loadtxt('../data/{}_ab.dat'.format(song_name))
        d["ab"] = d["smodel"].gen_alphabeta()
        for start, g in d["smodel"].gestures:
            d["ab"][start] = np.nan
        d["synth_ab"] = synth_ab
        nct = normalize_and_center(d["tutor"])
        param_feat= bsa.all_song_features(nct, d["sr"],
                                          freq_range=d["freq_range"],
                                          fft_step=d["fft_step"],
                                          fft_size=d["fft_size"])
        d["tfeat"] = get_features(d["tutor"], param_feat)
        d["smfeat"] = get_features(d["song"], param_feat)
        tmp = '../data/{}_out.wav'
        sr, d["synth"] = wavfile.read(tmp.format(song_name))
        d["Boari_score"] = utils.boari_synth_song_error(d["tutor"],
                                       d["synth"],
                                       d["run_param"]['coefs'],
                                       tutor_feat=param_feat)
        d["mtutor"] = bsa_measure(d["tutor"], d["sr"],
                             coefs=d["run_param"]['coefs'],
                             tutor_feat=param_feat)
        d["msynth"] = bsa_measure(d["synth"], d["sr"],
                             coefs=d["run_param"]['coefs'],
                             tutor_feat=param_feat)
        d["msong"] = bsa_measure(d["song"], d["sr"],
                            coefs=d["run_param"]['coefs'],
                            tutor_feat=param_feat)
        sim.append(d)
    return sim

def plot_gesture_starts(starts, scale=1):
    for start in starts:
        plt.axvline(start / scale, color="k", linewidth=1, alpha=0.2)

def plot_fig(sim, sims, titles):
    fnames = ["fm", "am", "entropy", "goodness", "amplitude", "pitch", "rms"]
    nb_row = 8 + len(fnames)
    nb_col = len(sim)
    color_song = "C1"
    color_synth = "C2"
    zoom = bsa.FFT_SIZE / bsa.FREQ_RANGE / 4
    
    plt.figure(figsize=(16, nb_row * 5))
    for i in range(nb_col):
        pos = 1 + i
        
        plt.subplot(nb_row, nb_col, pos)
        plt.plot(sim[i]["tutor"])
        plt.xlim(0, len(sim[i]["tutor"]))
        plt.title(sims[i]+"\n"+titles[i]+"\n\n"+"Tutor sound")
        pos += nb_col
        
        plt.subplot(nb_row, nb_col, pos)
        plt.plot(sim[i]["song"])
        plot_gesture_starts(sim[i]["starts"])
        plt.xlim(0, len(sim[i]["song"]))
        plt.title("Song model sound")
        pos += nb_col
    
        ax = plt.subplot(nb_row, nb_col, pos)
        bsa.spectral_derivs_plot(sim[i]["tspec"], contrast=0.01, ax=ax)
        ax.set_ylim(0, bsa.FREQ_RANGE * zoom)
        ax.set_title("Tutor spectral derivative")
        pos += nb_col
        
        ax = plt.subplot(nb_row, nb_col, pos)
        bsa.spectral_derivs_plot(sim[i]["smspec"], contrast=0.01, ax=ax)
        ax.set_ylim(0, bsa.FREQ_RANGE * zoom)
        plot_gesture_starts(sim[i]["starts"], scale=sim[i]["fft_step"])
        ax.set_title("Song spectral derivative")
        pos += nb_col
    
        ax = plt.subplot(nb_row, nb_col, pos)
        ax = utils.draw_learning_curve(sim[i]["rd"], ax=ax)
        ax.axhline(y=sim[i]["Boari_score"], color="C3",
                      linestyle='-', label="Boari's error")
        ax.legend()
        pos += nb_col
        
        for fname in fnames:
            plt.subplot(nb_row, nb_col, pos)
            plt.plot(sim[i]["tfeat"][fname], label="tutor")
            plt.plot(sim[i]["smfeat"][fname], label="song")
            plot_gesture_starts(sim[i]["starts"], scale=sim[i]["fft_step"])
            plt.xlim(0,len(sim[i]["tfeat"][fname]))
            plt.legend()
            plt.title(fname)
            pos += nb_col
            
        plt.subplot(nb_row, nb_col, pos)
        plt.plot(sim[i]["synth_ab"][:, 1], label="synth",color=color_synth)
        plt.plot(sim[i]["ab"][:, 1], label="song", color=color_song)
        plot_gesture_starts(sim[i]["starts"])
        plt.xlim(0,len(sim[i]["ab"][:, 1]))
        plt.legend()
        plt.title("Beta")
        pos += nb_col
        
        # Normalization of alpha values for better comparison
        num = sim[i]["synth_ab"][:, 0] - np.min(sim[i]["synth_ab"][:, 0])
        min_v = np.min(sim[i]["synth_ab"][:, 0])
        max_v = np.max(sim[i]["synth_ab"][:, 0])
        denum = max_v - min_v
        a_synth = num / denum
        num = sim[i]["ab"][:,0] - np.nanmin(sim[i]["ab"][:,0])
        denum = np.nanmax(sim[i]["ab"][:,0]) - np.nanmin(sim[i]["ab"][:,0])
        a_sm = num / denum

        # Inversion of the plot order for better readability
        ax = plt.subplot(nb_row, nb_col, pos)
        line1, = plt.plot(a_sm, label="song", color=color_song)
        line2, = plt.plot(a_synth, label="synth", color=color_synth)
        plot_gesture_starts(sim[i]["starts"])
        plt.xlim(0, len(a_sm))
        ax.legend((line2, line1), ("synth", "song"))
        plt.title("Alpha (normalized)")
        pos += nb_col
        
        # Calculation of each feature error
        amp, th = utils.carac_to_calculate_err_of_synth(sim[i]["synth"],
                                                        t_amp=sim[i]["tfeat"]["amplitude"])
        err_feat_vect = utils.err_per_feat(sim[i]["mtutor"],
                                           sim[i]["msong"])
        err_feat_vect_synth = utils.err_per_feat(sim[i]["mtutor"][amp > th],
                                                 sim[i]["msynth"][amp > th])
        x = np.arange(0,len(err_feat_vect))
        ax = plt.subplot(nb_row, nb_col, pos)
        synth_score = round(sim[i]["Boari_score"], 2)
        song_score = round(sim[i]["score"], 2)
        synth_label = "synth ({})".format(synth_score)
        song_label = "song ({})".format(song_score)
        plt.bar(x - 0.1, err_feat_vect_synth,
                width=0.2, align='center', label=synth_label, color=color_synth)
        plt.bar(x + 0.1, err_feat_vect,
                width=0.2, align='center', label=song_label, color=color_song)
        plt.xticks(x, fnames)
        shift = np.max(np.concatenate((err_feat_vect_synth, err_feat_vect))) / 100
        for index in x:
            v_synth = err_feat_vect_synth[index]
            v_song = err_feat_vect[index]
            ax.text(index - 0.1, v_synth + shift,
                    str(round(v_synth, 2)),
                    color=color_synth, ha="center", fontweight='bold')
            ax.text(index + 0.1, v_song + shift,
                    str(round(v_song, 2)),
                    color=color_song, ha="center", fontweight='bold')
        plt.legend()
        plt.title("Errors")

    plt.show()
