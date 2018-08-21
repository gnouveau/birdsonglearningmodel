"""Do Grid search with several configuration files."""

import argparse as ap
import os
from os.path import join, isdir
import shutil
import datetime
import json
from glob import glob
import logging
import itertools
import subprocess

import numpy as np
from scipy.io import wavfile
from joblib import Parallel, delayed, cpu_count

from song_fitter import fit_song, COMP_METHODS
from datasaver import DataSaver

logger = logging.getLogger('root')
EDITOR = os.environ.get('EDITOR', 'vim')


def get_confs(confdir):
    """Iterator over all combinations of the files in subfolders.
    Be careful of the .json~ files in the config folders """
    confs = []
    for folder in sorted(glob(join(confdir, "*"))):
        if not isdir(folder):
            continue
        confs.append([])
        for conf_file_name in sorted(glob(join(folder, '*.json'))):
            with open(conf_file_name) as conf_file:
                try:
                    confs[-1].append(json.load(conf_file))
                except json.decoder.JSONDecodeError:
                    print('json.decoder.JSONDecodeError:', conf_file_name)
                    raise
    for conf_prod in itertools.product(*confs):
        tot_conf = dict()
        names = []
        for conf in conf_prod:
            tot_conf.update(conf)
            try:
                names.append(conf['name'])
            except KeyError:
                pass
            
        yield '+'.join(names), tot_conf


def start_run(run_name, conf, res_grid_path):
    """Start a run with run_name and the conf `conf`."""
    try:
        start = datetime.datetime.now()
        print('starting {}'.format(run_name))
        
        if conf['seed'] is not None:
            conf['rng_obj'] = np.random.RandomState(conf['seed'])
        else:
            conf['rng_obj'] = np.random.RandomState()
        
        conf['comp_obj'] = COMP_METHODS[conf.get('comp', 'linalg')]
        conf['name'] = run_name
        with open(conf['tutor'], 'rb') as tutor_f:
            sr, tutor = wavfile.read(tutor_f)
        run_path = join(res_grid_path, run_name)
        os.makedirs(run_path)
        shutil.copyfile(conf['tutor'], join(run_path, 'tutor.wav'))
        datasaver = DataSaver(join(run_path, 'data_cur.pkl'))
        with open(join(run_path, 'conf.json'), 'w') as conf_file:
            json.dump({key: conf[key] for key in conf
                       if not key.endswith('obj')}, conf_file, indent=4)
        fit_song(tutor, conf, datasaver)
        datasaver.write(join(run_path, 'data.pkl'))
        print(run_name, 'is over and took', datetime.datetime.now() - start)
        print('By the way, it is {}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    except Exception as e:
        logger.error('{} crashed'.format(run_name))
        logger.exception(e)


def main():
    """Do the gridsearch over several configuration files."""
    parser = ap.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('--cpu', type=int, default=cpu_count()//2)
    parser.add_argument('--single-job', action="store_true")
    parser.add_argument('--outdir', type=str)
    parser.add_argument('confdir', type=str)
    parser.add_argument('--no-desc', dest='edit_desc', action='store_false')
    args = parser.parse_args()
    
    # /!\ If single-job is used, outdir has to be defined
    if args.single_job:
        with open(args.confdir, 'r') as f:
            conf = json.load(f)
        start_run(args.name, conf, args.outdir)
    else:
        start = datetime.datetime.now()
        packed_run_name = '{}_{}'.format(args.name,
                                         start.strftime('%y%m%d_%H%M%S'))
        res_grid_path = join('res', packed_run_name)
        os.makedirs(res_grid_path)
        shutil.copy('desc.template.md', join(res_grid_path, 'desc.md'))
        if args.edit_desc:
            subprocess.call([EDITOR, join(res_grid_path, 'desc.md')])
        print('Using {} cpu'.format(args.cpu))
        logging.basicConfig(level=logging.WARNING)
        Parallel(n_jobs=args.cpu)(
            delayed(start_run)(run_name, conf, res_grid_path)
            for run_name, conf in get_confs(args.confdir))
        logger.info('All over!')
        print('took', datetime.datetime.now() - start)


if __name__ == "__main__":
    main()
