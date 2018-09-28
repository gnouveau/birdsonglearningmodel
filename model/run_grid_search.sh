#!/bin/sh
#PBS -N new_params_low_div_th_part_2
#PBS -l walltime=320:00:00
#PBS -l nodes=1:ppn=32
#PBS -d /home/gilles.nouveau/birdsonglearningmodel/model

sim_name="long_div_th_new_params"

echo "*** start cluster_simulation ***"
/home/gilles.nouveau/.virtualenvs/birdsong/bin/python grid_search.py --name $sim_name --cpu 32 --no-desc confs/$sim_name
echo "*** cluster_simulation over ***"

#echo "*** start one simulation ***"
#/home/gilles.nouveau/.virtualenvs/birdsong/bin/python song_fitter.py --config confs/$sim_name.json --no-desc
#echo "*** one simulation over ***"
