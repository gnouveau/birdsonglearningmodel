"""Train a model to reproduce a specific birdsong.

This module can be called from the shell. It takes a configuration file as a
parameters. Some parameters can be directly overwritten by the script call.

This module also contains the function `fit_song` which is the main function
of this research project. See `fit_song` documentation for a full description.

Example
-------

Run the algorithm with the parameters from a file in JSON format.

    $ python song_fitter.py --config confs/conf.json

Display help to see all the argument that can be given to the song fitter
script.

    $ python song_fitter.py --help

Prevent the editor call to take notes before the run.

    $ python song_fitter.py --config confs/conf.json --no-desc

"""
# Resolve a problem between matplotlib, tkinter and python virtualenv
import sys
if "matplotlib" not in sys.modules:
    import matplotlib
    matplotlib.use('agg')
    
import argparse as ap
import logging
import os
import datetime
import json
import subprocess
from pprint import pformat
from shutil import copyfile
from subprocess import call

import numpy as np
from fastdtw import fastdtw
from scipy.io import wavfile

from datasaver import DataSaver, QuietDataSaver
from day_optimisers import optimise_gesture_dummy, optimise_gesture_padded,\
                           optimise_gesture_whole,\
                           optimise_gesture_whole_local_search,\
                           optimise_proportional_training,\
                           optimise_root_mean_square_error
from measures import bsa_measure, get_scores, normalize_and_center
from night_optimisers import mutate_best_models_dummy, \
                             mutate_best_models_elite, \
                             mutate_microbial, \
                             mutate_microbial_extended_elite, \
                             mutate_microbial_extended_uniform, \
                             mutate_microbial_diversity_uniform, \
                             mutate_microbial_diversity_continuous_uniform, \
                             mutate_microbial_diversity_distance_uniform
from song_model import SongModel
import birdsonganalysis as bsa

logger = logging.getLogger('root')
EDITOR = os.environ.get('EDITOR', 'vim')

"""
Available day learning models for the configuration files
"""
DAY_LEARNING_MODELS = {
    'optimise_gesture_dummy': optimise_gesture_dummy,
    'optimise_gesture_padded': optimise_gesture_padded,
    'optimise_gesture_whole': optimise_gesture_whole,
    'optimise_gesture_whole_local_search': optimise_gesture_whole_local_search,
    'optimise_proportional_training': optimise_proportional_training,
    'optimise_root_mean_square_error': optimise_root_mean_square_error
}
"""
Available night learning models for the configuration files
"""
NIGHT_LEARNING_MODELS = {
    'mutate_best_models_dummy': mutate_best_models_dummy,
    'mutate_best_models_elite': mutate_best_models_elite,
    'mutate_microbial': mutate_microbial,
    'mutate_microbial_extended_elite': mutate_microbial_extended_elite,
    'mutate_microbial_extended_uniform': mutate_microbial_extended_uniform,
    'mutate_microbial_diversity_uniform': mutate_microbial_diversity_uniform,
    'mutate_microbial_diversity_continuous_uniform': mutate_microbial_diversity_continuous_uniform,
    'mutate_microbial_diversity_distance_uniform': mutate_microbial_diversity_distance_uniform,
    'no_night': None
}
"""
Available comparison methods for the configuration files
"""
# TODO: fastdtw use still not fully implemented
COMP_METHODS = {'linalg': lambda g, c: np.linalg.norm(g - c),
                'fastdtw': lambda g, c: fastdtw(g, c, dist=2, radius=1)[0]}

def fit_song(tutor_song, conf, datasaver=None):
    """Fit a song with a day and a night phase.

    This function returns a list of SongModel.

    The fit is split in two phases: A day part and a night part. The day part
    is a simple optimisation algorithm within gesture. The night part
    is a restructuring algorithm. See details in the modules
    `song_model.SongModel`, `day_optimisers` and `night_optimisers`


    Parameters
    ----------
    tutor_song : 1D array_like
        The tutor song that the algorithm will try to reproduce.
        It will be normalized between -1 and +1 internally.
        You don't need to do it yourself.
    conf : dict
        The dictionnary of all the parameters needed for the run.
        Values that are required with `fit_song are`:
            'dlm': The day learning model key from DAY_LEARNING_MODELS dict.
            'nlm': The night learning model key from NIGHT_LEARNING_MODELS dict.
            'days': The number of day for a run
            'concurrent': The number of concurrent songs during the day.
            'comp_obj': a callable for the comparison.
            'rng_obj': a `numpy.RandomState` object for the random generation.
            'measure_obj': a callable to measure song features.

        'comp_obj', 'rng_obj' and 'measure_obj' are not importable from json
        files, but can be built easily by reading arguments like 'seed' or keys
        from the configuration files, like 'dlm' and 'nlm'.

        The required values depend on the day learning model and night
        learning model picked.

    Returns
    -------
    songmodels : List[SongModel]
        The songmodels at the end of the training.

    See also
    --------
    song_model.SongModel
    day_optimisers
    night_optimisers

    """
    tutor_song = normalize_and_center(tutor_song)
    
    tutor_feat = bsa.all_song_features(tutor_song, bsa.SR,
                                       freq_range=bsa.FREQ_RANGE,
                                       fft_step=bsa.FFT_STEP,
                                       fft_size=bsa.FFT_SIZE)
    
    conf['measure_obj'] = lambda x: bsa_measure(x, bsa.SR, 
                                                coefs=conf['coefs'],
                                                tutor_feat=tutor_feat) 
    
    day_optimisation = DAY_LEARNING_MODELS[conf['dlm']]
    night_optimisation = NIGHT_LEARNING_MODELS[conf['nlm']]
    nb_day = conf['days']
    nb_conc_song = conf['concurrent']
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    rng = conf['rng_obj']
    nb_split = conf.get('split', 10)
    # muta_proba is a list of 4 values: [P(deletion), P(division), P(movement), P(no_mutation)]
    muta_proba = conf['muta_proba']

    songs = [SongModel(song=tutor_song, priors=conf['prior'],
                       nb_split=nb_split, rng=rng, muta_proba=muta_proba)
             for i in range(nb_conc_song)]
    
    goal = measure(tutor_song)
    
    if datasaver is None:
        datasaver = QuietDataSaver()
    datasaver.add(moment='Start', songs=songs,
                  scores=get_scores(goal, songs, measure, comp))

    cond1 = conf['dlm'] == 'optimise_gesture_whole'
    cond2 = conf['dlm'] == 'optimise_gesture_whole_local_search'
    cond3 = conf['dlm'] == 'optimise_proportional_training'
    if cond1 or cond2 or cond3:
        target = goal
    else:
        # case where conf['dlm'] == 'optimise_root_mean_square_error'
        target = tutor_song

    for iday in range(nb_day):
        logger.info('*\t*\t*\tDay {} of {}\t*\t*\t*'.format(iday+1, nb_day))
        with datasaver.set_context('day_optim'):
            songs = day_optimisation(songs, target, conf,
                                     datasaver=datasaver, iday=iday)
        score = get_scores(goal, songs, measure, comp)
        if iday + 1 != nb_day:
            logger.debug(score)
            datasaver.add(moment='before_night',
                          songs=songs, scores=score)
            logger.info('z\tz\tz\tNight\tz\tz\tz')
            with datasaver.set_context('night_optim'):
                if conf['nlm'] == "no_night":
                    pass
                elif conf['nlm'] == "mutate_microbial_diversity_distance_uniform":
                    songs = night_optimisation(songs, conf, i_night=iday,
                                               datasaver=datasaver)
                elif conf['nlm'] == "mutate_microbial_diversity_continuous_uniform":
                    songs = night_optimisation(songs, conf, i_night=iday,
                                               datasaver=datasaver)
                elif conf['nlm'] == "mutate_microbial_diversity_uniform":
                    songs = night_optimisation(songs,
                                               goal, iday, nb_day, conf, 
                                               datasaver=datasaver)
                else:
                    songs = night_optimisation(songs,
                                               goal,
                                               conf, 
                                               datasaver=datasaver)
            score = get_scores(goal, songs, measure, comp)
            datasaver.add(moment='after_night', songs=songs, scores=score)
        datasaver.write()
    datasaver.add(moment='End', songs=songs,
                  scores=get_scores(goal, songs, measure, comp))
    return songs


def get_git_revision_hash():
    """Get the git commit/revision hash.

    Knowing the git revision hash is helpful to reproduce a result with
    the code corresponding to a specific run.

    """
    try:
        return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']),
                   'utf8').strip()
    except OSError:
        return None


def main():
    """Main function for this module, called if not imported."""
    global NIGHT_LEARNING_MODELS, DAY_LEARNING_MODELS, COMP_METHODS
    logging.basicConfig(level=logging.DEBUG)

    start = datetime.datetime.now()
    tsong = None
    parser = ap.ArgumentParser(
        description="""
        reproduce the learning of a zebra finch for a given tutor song.
        """
    )
    # if tutor not defined in the command-line, it gets the value None
    parser.add_argument('tutor', type=ap.FileType('rb'), nargs='?',
                        help='The targeted song to learn')
    parser.add_argument('--config', type=ap.FileType('r'), required=False,
                        help='The config file to take the parameters from.')
    parser.add_argument('-d', '--days', type=int, required=None,
                        help='number of days')
    parser.add_argument('-t', '--train-per-day', type=int, required=False,
                        help='number of training for a gesture per day')
    parser.add_argument('-c', '--concurrent', type=int, required=False,
                        help='number of concurrent model for the song')
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='name of the trial for logging')
    parser.add_argument('-s', '--seed', type=int, required=False,
                        help='seed for the random number generator')
    parser.add_argument('-r', '--replay', type=int, required=False,
                        help='number of passes for new generations during'
                        ' night')
    parser.add_argument('-i', '--iter-per-train', type=int, required=False,
                        help='number of iteration when training a gesture')
    parser.add_argument('--comp', type=str, required=False, default='linalg',
                        choices=COMP_METHODS,
                        help='comparison method to use')
    parser.add_argument('--dlm', type=str, required=False,
                        choices=DAY_LEARNING_MODELS, help="day learning model")
    parser.add_argument('--nlm', type=str, required=False,
                        choices=NIGHT_LEARNING_MODELS,
                        help="night learning model")
    # edit_conf = False by default. If --edit-conf appears in the command-line, becomes True
    parser.add_argument('--edit-conf', action='store_true')
    parser.add_argument('--coefs', type=ap.FileType('r'),
                        default='confs/default_coefs.json',
                        help="file with the coefs")
    parser.add_argument('--priors', type=ap.FileType('r'),
                        default="confs/default_prior_dev.json")
    # if --no-desc appears in the command-line, it gets False, else it's True
    parser.add_argument('--no-desc', dest='edit_desc', action='store_false')
    args = parser.parse_args()
    if args.seed is None:
        seed = int(datetime.datetime.now().timestamp())
    else:
        seed = args.seed
    rng = np.random.RandomState(seed)
    conf = {}
    # if the --config option is defined, meaning a config file is used
    if args.config:
        conf.update(json.load(args.config))
        try:  # Warning if reproduction (with commit key) and different commits
            if conf['commit'] != get_git_revision_hash():
                logger.warning('Commit recommended for the conf is different'
                               ' from the current commit.')
        except KeyError:
            pass
        try:
            # sr = sampling rate
            sr, tsong = wavfile.read(conf['tutor'])
        except KeyError:
            pass
        try:
            seed = conf['seed']
            rng = np.random.RandomState(seed)
        except KeyError:
            pass
        conf['commit'] = get_git_revision_hash()
    argdata = {'days': args.days,
               'train_per_day': args.train_per_day,
               'concurrent': args.concurrent,
               'name': args.name,
               'seed': seed,
               'replay': args.replay,
               'iter_per_train': args.iter_per_train,
               'commit': get_git_revision_hash(),
               'comp': args.comp,
               'dlm': args.dlm,
               'nlm': args.nlm}
    if args.tutor is not None:
        argdata['tutor'] = args.tutor.name
    if tsong is None:
        sr, tsong = wavfile.read(args.tutor)

    # the parameters in the command-line overwrite the ones defined with --config
    conf.update({k: v for k, v in argdata.items() if v is not None})

    date = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    run_name = '{}_{}'.format(conf['name'], date)
    path = 'res/{}'.format(run_name)
    os.makedirs(path)
    wavfile.write(os.path.join(path, 'tutor.wav'), sr, tsong)
    prior_dev = json.load(args.priors)
    
    # update values, if they were not already defined in the config file
    for key, value in prior_dev.items():
        if key not in conf:
            conf[key] = value
            
    if 'coefs' not in conf:
        coefs = json.load(args.coefs)
        conf.update(coefs)
    if args.edit_desc:
        write_run_description(path)
    with open(os.path.join(path, 'conf.json'), 'w') as f:
        json.dump({k: conf[k] for k in conf if not k.endswith('_obj')},
                  f, indent=4)  # human readable parameters
    if args.edit_conf:
        call([EDITOR, os.path.join(path, 'conf.json')])
        with open(os.path.join(path, 'conf.json'), 'r') as f:
            conf = json.load(f)

    datasaver = DataSaver(defaultdest=os.path.join(path, 'data_cur.pkl'))
    logger.info(pformat(conf))

    conf['rng_obj'] = rng
    conf['comp_obj'] = COMP_METHODS[conf['comp']]

    #########################################
    # STOP READING CONF; START THE LEARNING #
    #########################################
    try:
        fit_song(tsong, conf, datasaver=datasaver)
    except KeyboardInterrupt:
        logger.warning('Aborted')
        with open(os.path.join(path, 'aborted.txt'), 'a') as f:
            f.write('aborted\n')
    finally:
        logger.info('Saving the data.')
        datasaver.write(os.path.join(path, 'data.pkl'))
    logger.info('!!!! Learning over !!!!')
    try:
        subprocess.Popen(['notify-send',
                          '{} is finished'.format(run_name)])
    except OSError:
        pass
    total_time = datetime.datetime.now() - start
    string = 'Run {} is over. Took {}'.format(run_name, total_time)
    logger.info(string)
    with open(os.path.join(path, 'execution_time.txt'), 'a') as f:
        f.write(string)


def write_run_description(path):
    """Open an editor with a prefilled file to describe the run."""
    copyfile('desc.template.md', os.path.join(path, 'desc.md'))
    call([EDITOR, os.path.join(path, 'desc.md')])


if __name__ == '__main__':
    main()
