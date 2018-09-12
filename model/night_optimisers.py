"""Collection of functions to optimise songs during the night.

These algorithms are mainly restructuring algorithms.
"""
import logging

import numpy as np

from datasaver import QuietDataSaver
from measures import (get_scores, genetic_neighbours,
                      generate_mat_neighbours_metrics,
                      update_mat_neighbours_metrics,
                      generate_mat_neighbours_distances,
                      update_mat_neighbours_distances
                      )

logger = logging.getLogger('night_optimisers')


def rank(array):
    """Give the rank of each element of an array.

    >>> rank([3 5 2 6])
    [2 3 1 4]

    Indeed, 2 is the 1st smallest element of the array, 3 is the 2nd smallest,
    and so on.
    """
    temp = np.argsort(array)
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(1, len(array) + 1)
    return ranks


def mutate_best_models_dummy(songs, goal, measure, comp, nb_replay,
                             datasaver=None, rng=None):
    """Dummy selection and mutation of the best models."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    nb_conc_song = len(songs)
    night_songs = np.array(songs)
    pscore = get_scores(goal, songs, measure, comp)
    nb_conc_night = nb_conc_song * 2
    fitness = len(night_songs) - rank(pscore)
    night_songs = np.random.choice(night_songs, size=nb_conc_night,
                                   p=fitness/np.sum(fitness))
    night_songs = np.array([song.mutate(nb_replay) for song in night_songs])
    score = get_scores(goal, night_songs, measure, comp)
    fitness = len(night_songs) - rank(score)
    isongs = rng.choice(len(night_songs),
                        size=nb_conc_song, replace=False,
                        p=fitness/np.sum(fitness))
    nsongs = night_songs[isongs].tolist()
    datasaver.add(prev_songs=songs, prev_scores=pscore, new_songs=nsongs,
                  new_scores=score[isongs])
    return nsongs


def mutate_best_models_elite(songs, goal, conf,
                             datasaver=None):
    """Elite selection and mutation of the best models.

    Keep the best mutations after each replay, parents are present in the
    selection.
    """
    if datasaver is None:
        datasaver = QuietDataSaver()
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_replay = conf['replay']
    rng = conf['rng_obj']
    nb_conc_song = len(songs)
    pscore = get_scores(goal, songs, measure, comp)
    score = pscore
    nb_conc_night = nb_conc_song * 2
    # make night_songs an array to do list indexes.
    night_songs = np.array(songs)
    for dummy_i in range(nb_replay):
        fitness = len(night_songs) - rank(score)
        night_songs = np.random.choice(night_songs, size=nb_conc_night,
                                       p=fitness/np.sum(fitness))
        night_songs = np.array([song.mutate() for song in night_songs])
        score = get_scores(goal, night_songs, measure, comp)
        fitness = len(night_songs) - rank(score)
        isongs = rng.choice(len(night_songs),
                            size=nb_conc_song, replace=False,
                            p=fitness/np.sum(fitness))
        night_songs = night_songs[isongs]
        score = score[isongs]

    nsongs = night_songs.tolist()
    datasaver.add(prev_songs=songs, prev_scores=pscore, new_songs=nsongs,
                  new_scores=score)
    return nsongs


def mutate_microbial(songs, goal, conf, datasaver=None):
    """Microbial GA implementation for the songs."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    songs = np.asarray(songs)
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_replay = conf['replay']
    rng = conf['rng_obj']
    for i in range(nb_replay):
        picked_songs = rng.choice(len(songs), size=2, replace=False)
        scores = get_scores(goal, songs[picked_songs], measure, comp)
        best = np.argmin(scores)
        loser = 1 - best  # if best = 0, loser = 1, else: loser = 0
        songs[picked_songs[loser]] = songs[picked_songs[best]].mutate()
    return songs


def mutate_microbial_diversity(songs, goal, cur_day, nb_day,
                               conf, datasaver=None):
    """Microbial GA implementation for the songs."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    songs = np.asarray(songs)
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_replay = conf['replay']
    rng = conf['rng_obj']
    threshold = conf.get('diversity_threshold', 2000)
#    bloat_weight = conf.get('bloat_weight', 0)
    diversity_weight = conf.get('diversity_weight', 1)
#    diversity_decay = conf.get('decay', None)
#    if diversity_decay == 'linear':
#        diversity_weight = diversity_weight * (1 - (cur_day / (nb_day - 1)))
    datasaver.add(cond="init_pop", songs=songs)
    for i in range(nb_replay):
        picked_songs = rng.choice(len(songs), size=2, replace=False)
        scores = get_scores(goal, songs[picked_songs], measure, comp)
        nb_similar = genetic_neighbours(songs[picked_songs], songs, threshold)
        
        # TODO: to choose:  if used or not
        # Penalise if too much gestures
#        nb_gestures = np.array([len(songs[picked_songs[0]].gestures),
#                                len(songs[picked_songs[1]].gestures)])
#        best_id = np.argmin(scores * (nb_similar**diversity_weight)
#                         * (nb_gestures**bloat_weight))
        
        best_id = np.argmin(scores * (nb_similar**diversity_weight))
        
        loser_id = 1 - best_id  # if best_id == 0: loser_id = 1, else: loser_id = 0
        new_song = songs[picked_songs[best_id]].mutate()
        songs[picked_songs[loser_id]] = new_song
        datasaver.add(cond="after_a_replay", i_night=cur_day, i_replay=i,
                      i_loser=picked_songs[loser_id],
                      i_winner=picked_songs[best_id],
                      new_song=new_song)
    return songs


def mutate_microbial_diversity_continuous(songs, conf, i_night=None,
                                          datasaver=None):
    """microbial genetic algorithm.
    fitness objective: pressure for diversity: maximize the mean metric of
    'distance' from the others songs"""
    if datasaver is None:
        datasaver = QuietDataSaver()
    songs = np.asarray(songs)
    nb_replay = conf['replay']
    rng = conf['rng_obj']
    mat_neigh_metric = generate_mat_neighbours_metrics(songs)
    datasaver.add(cond="init_pop", songs=songs,
                  mat_neigh_metric=mat_neigh_metric)
    for i in range(nb_replay):
        picked_songs = rng.choice(len(songs), size=2, replace=False)
        best_id = np.argmax(np.mean(mat_neigh_metric[picked_songs], axis=1))
        loser_id = 1 - best_id
        new_song = songs[picked_songs[best_id]].mutate()
        songs[picked_songs[loser_id]] = new_song
        mat_neigh_metric = update_mat_neighbours_metrics(mat_neigh_metric,
                                                         picked_songs[loser_id],
                                                         songs)
        datasaver.add(cond="after_a_replay", i_night=i_night, i_replay=i,
                      i_loser=picked_songs[loser_id],
                      i_winner=picked_songs[best_id],
                      new_song=new_song,
                      mat_neigh_metric=mat_neigh_metric)
    return songs


def mutate_microbial_diversity_distance(songs, conf, i_night=None,
                                        datasaver=None):
    """microbial genetic algorithm.
    fitness objective: pressure for diversity: maximize the mean distance
    from the others songs"""
    if datasaver is None:
        datasaver = QuietDataSaver()
    songs = np.asarray(songs)
    nb_replay = conf['replay']
    rng = conf['rng_obj']
    mat_neigh_dist = generate_mat_neighbours_distances(songs)
    datasaver.add(cond="init_pop", songs=songs,
                  mat_neigh_dist=mat_neigh_dist)
    for i in range(nb_replay):
        picked_songs = rng.choice(len(songs), size=2, replace=False)
        best_id = np.argmax(np.mean(mat_neigh_dist[picked_songs], axis=1))
        loser_id = 1 - best_id
        new_song = songs[picked_songs[best_id]].mutate()
        songs[picked_songs[loser_id]] = new_song
        mat_neigh_dist = update_mat_neighbours_distances(mat_neigh_dist,
                                                         picked_songs[loser_id],
                                                         songs)
        datasaver.add(cond="after_a_replay", i_night=i_night, i_replay=i,
                      i_loser=picked_songs[loser_id],
                      i_winner=picked_songs[best_id],
                      new_song=new_song,
                      mat_neigh_dist=mat_neigh_dist)
    return songs


#def mutate_diversity_multi_criteria(songs, goal, conf, i_night=None,
#                                    datasaver=None):
#    if datasaver is None:
#        datasaver = QuietDataSaver()
#    songs = np.asarray(songs)
#    nb_replay = conf['replay']
#    rng = conf['rng_obj']
#    measure = conf['measure_obj']
#    comp = conf['comp_obj']
#    score_weight, dist_weight = conf.get('criteria_weight', [0.5, 0.5])
#    mat_neigh_dist = generate_mat_neighbours_distances(songs)
#    datasaver.add(cond="init_pop", songs=songs,
#                  mat_neigh_dist=mat_neigh_dist)
#    for i in range(nb_replay):
#        picked_songs = rng.choice(len(songs), size=2, replace=False)
#        scores = get_scores(goal, songs[picked_songs], measure, comp)
#        distances = np.mean(mat_neigh_dist[picked_songs], axis=1)
#        """
#        TODO: un calcul malin utilisant le score et la diversité (multicritère)
#        par ex, somme pondérée, aggrégation des 2 critères, pareto dominance, etc...
#        """
#        # calcul pas malin
#        best_id = np.argmax(scores * distances)
#        loser_id = 1 - best_id
#        songs[picked_songs[loser_id]] = songs[picked_songs[best_id]].mutate()
#        mat_neigh_dist = update_mat_neighbours_distances(mat_neigh_dist,
#                                                         picked_songs[loser_id],
#                                                         songs)
#        datasaver.add(cond="after_a_replay", i_night=i_night, i_replay=i,
#                      i_loser=picked_songs[loser_id],
#                      i_winner=picked_songs[best_id],
#                      songs=songs,
#                      mat_neigh_dist=mat_neigh_dist)
#    return songs


def extend_pop(songs, conf, datasaver=None):
    """Extend the size of a population."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    new_pop_size = conf['night_concurrent']
    rng = conf['rng_obj']
    songs = np.asarray(songs)
    night_pop = rng.choice(songs, size=new_pop_size, replace=True)
    night_pop = np.array([song.mutate() for song in night_pop])
    return night_pop


def restrict_pop_elite(songs, goal, conf, datasaver=None):
    """Restrict the size of a population with elitism (bests are kept)."""
    nb_concurrent = conf['concurrent']
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    indices = np.argpartition(get_scores(goal, songs, measure, comp),
                              -nb_concurrent)[-nb_concurrent:]
    return np.asarray(songs[indices])


def restrict_pop_uniform(songs, conf, datasaver=None):
    """Restrict the size of a population by uniform random selection."""
    nb_concurrent = conf['concurrent']
    rng = conf['rng_obj']
    return rng.choice(songs, nb_concurrent, replace=False)


def restrict_pop_rank(songs, goal, conf, datasaver=None):
    """Restrict the size of a population by rank driven random selection."""
    nb_concurrent = conf['concurrent']
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    rng = conf['rng_obj']
    score = get_scores(goal, songs, measure, comp)
    fitness = len(songs) - rank(score)
    return rng.choice(songs, nb_concurrent, replace=False,
                      p=fitness/np.sum(fitness))


def mutate_microbial_extended_elite(songs, goal, conf, datasaver=None):
    """Do a microbial on an extended population and restrict with elitism."""
    datasaver.add(label='night', cond='before_evening', pop=songs)
    new_pop = extend_pop(songs, conf, datasaver)
    datasaver.add(label='night', cond='evening', pop=new_pop)
    mutate_pop = mutate_microbial(new_pop, goal, conf, datasaver)
    datasaver.add(label='night', cond="before_morning", pop=mutate_pop)
    new_pop = restrict_pop_elite(mutate_pop, goal, conf, datasaver)
    datasaver.add(label='night', cond='morning', pop=new_pop)
    return new_pop


def mutate_microbial_extended_uniform(songs, goal, conf, datasaver=None):
    """Do a microbial on an extended population and restrict by random."""
    datasaver.add(label='night', cond='before_evening', pop=songs)
    new_pop = extend_pop(songs, conf, datasaver)
    datasaver.add(label='night', cond='evening', pop=new_pop)
    mutate_pop = mutate_microbial(new_pop, goal, conf, datasaver)
    datasaver.add(label='night', cond="before_morning", pop=mutate_pop)
    new_pop = restrict_pop_uniform(mutate_pop, conf, datasaver)
    datasaver.add(label='night', cond='morning', pop=new_pop)
    return new_pop


def mutate_microbial_diversity_uniform(songs, goal, cur_day, nb_day,
                                       conf, datasaver=None):
    """Do a microbial on an extended population and restrict by random.
    The selection in the microbial genetic algorithm penalize
    songs which have more neighbours"""
    datasaver.add(label='night', cond='before_evening', pop=songs)
    new_pop = extend_pop(songs, conf, datasaver)
    datasaver.add(label='night', cond='evening', pop=new_pop)
    mutate_pop = mutate_microbial_diversity(new_pop, goal, cur_day,
                                            nb_day, conf, datasaver)
    datasaver.add(label='night', cond="before_morning", pop=mutate_pop)
    new_pop = restrict_pop_uniform(mutate_pop, conf, datasaver)
    datasaver.add(label='night', cond='morning', pop=new_pop)
    return new_pop


def mutate_microbial_diversity_continuous_uniform(songs, conf, i_night=None,
                                                  datasaver=None):
    """Do a microbial on an extended population and restrict by random.
    The selection in the microbial genetic algotrithm favours
    song which have the biggest mean metric of 'distance'
    from the others songs"""
    datasaver.add(label='night', cond='before_evening', pop=songs)
    new_pop = extend_pop(songs, conf, datasaver)
    datasaver.add(label='night', cond='evening', pop=new_pop)
    mutate_pop = mutate_microbial_diversity_continuous(new_pop, conf, i_night,
                                                       datasaver)
    datasaver.add(label='night', cond="before_morning", pop=mutate_pop)
    new_pop = restrict_pop_uniform(mutate_pop, conf, datasaver)
    datasaver.add(label='night', cond='morning', pop=new_pop)
    return new_pop

def mutate_microbial_diversity_distance_uniform(songs, conf, i_night=None,
                                                datasaver=None):
    """Do a microbial on an extended population and restrict by random.
    The selection in the microbial genetic algotrithm favours
    song which have the biggest mean distance
    from the others songs"""
    datasaver.add(label='night', cond='before_evening', pop=songs)
    new_pop = extend_pop(songs, conf, datasaver)
    datasaver.add(label='night', cond='evening', pop=new_pop)
    mutate_pop = mutate_microbial_diversity_distance(new_pop, conf, i_night,
                                                       datasaver)
    datasaver.add(label='night', cond="before_morning", pop=mutate_pop)
    new_pop = restrict_pop_uniform(mutate_pop, conf, datasaver)
    datasaver.add(label='night', cond='morning', pop=new_pop)
    return new_pop
