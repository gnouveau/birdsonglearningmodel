"""Define the SongModel class."""

from copy import deepcopy

import numpy as np
import logging
from synth import only_sin, gen_alphabeta, synthesize
import birdsonganalysis as bsa


logger = logging.getLogger('songmodel')


class SongModel:
    """Song model structure."""

    def __init__(self, song, gestures=None, nb_split=20, rng=None,
                 parent=None, priors=None, muta_proba=None):
        """
        Initialize the song model structure.

        GTE - list of the GTE of the song
        priors - list of the priors of the song for a gesture
        """
        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)
        self.song = song
        if gestures is None:
            if priors is None:
                raise ValueError('should give prior if no gestures.')
            gestures = [[start, np.array(priors)]
                        for start in sorted([0] +
                        list(self.rng.randint(100, len(song) - 100,
                                         size=nb_split-1)))]
            # remove gestures that are too close (or equal)
            remove_gesture = True
            i_start = 0
            while remove_gesture:
                remove_gesture = False
                for i in range(i_start, len(gestures)-1):
                    if gestures[i+1][0] - gestures[i][0] < 100:
                        # deletion in the list looked with the loop
                        # it's ok because we break afterwards
                        del gestures[i+1]
                        remove_gesture = True  # need to remove more stuff
                        i_start = i
                        break
        self.gestures = deepcopy(gestures)
        # Do not keep track of parent for now, avoid blow up in copy
        self.parent = None

        if muta_proba is None:
            string = 'Have to define the probabilities of deletion, division and movement'
            raise Exception(string)
        if len(muta_proba) != 4:
            string = 'The list of mutation probabilities has to have 4 values: '
            string += 'P(deletion), P(division), P(movement) and P(no_mutation)'
            raise Exception(string)
        if sum(muta_proba) != 1:
            raise Exception('Sum of probabilites is not equal to 1 ({})'.format(muta_proba))
        # Comment: the last value of muta_proba will not be used in practice
        self.muta_proba = muta_proba
        self.cum_sum_proba = [sum(muta_proba[:end]) for end in range(1, len(muta_proba)+1)]


    def mutate(self, n=1):
        """Give a new song model with new GTEs.
        n: number of mutations
        """
        gestures = deepcopy(self.gestures)
        for i in range(n):
            act = self.rng.uniform()
            if act <= self.cum_sum_proba[0] and len(gestures) > 2:  # Delete a gesture
                logger.info('deleted')
                to_del = self.rng.randint(len(gestures))
                del gestures[to_del]
                if to_del == 0:  # if the first gesture is suppressed
                    gestures[0][0] = 0  # the new one has to start at 0
            elif act <= self.cum_sum_proba[1]:  # split one gesture into two. Create a new gesture
                add_after = self.rng.randint(len(gestures) - 1)
                try:
                    add_at = self.rng.randint(gestures[add_after][0] + 100,
                                           gestures[add_after + 1][0] - 100)
                except ValueError:  # There is no new place
                    continue
                logger.info('split')
                new_gesture = self.shift_gesture(gestures[add_after], add_at)
                gestures.insert(add_after + 1, new_gesture)
            elif act <= self.cum_sum_proba[2]:  # change the gesture's start
                logger.info('moved')
                to_move = self.rng.randint(1, len(gestures))
                min_pos = gestures[to_move - 1][0] + 100
                try:
                    max_pos = gestures[to_move + 1][0] - 100
                except IndexError:  # Perhaps we have picked the last gesture
                    max_pos = len(self.song) - 100
                cur_pos = gestures[to_move][0]
                if cur_pos < min_pos or max_pos < cur_pos:  # not enough space
                    logger.error('to_move < min_pos or max_pos < to_move')
                else:
                    try:
                        new_pos = self.rng.triangular(min_pos, cur_pos, max_pos)  # triangular distribution
                    except Exception as e:
                        logger.error("song_model.mutate(): triangular function, exception raised: {}. So no mutation".format(e))
                        continue
                    new_pos = int(new_pos)
                    gestures[to_move] = self.shift_gesture(gestures[to_move], new_pos)
            else:  # Do not mutate
                logger.info('no mutation')
                pass
            # clean GTEs
            gestures.sort(key=lambda x: x[0])
            clean = False
            i_start = 1
            while not clean:
                for i in range(i_start, len(gestures)):
                    if gestures[i][0] - gestures[i - 1][0] < 100:
                        del gestures[i]
                        i_start = i
                        break
                else:  # If there is no break (for/else python syntax)
                    clean = True
            if len(self.song) - gestures[-1][0] < 100:
                del gestures[-1]
        return SongModel(self.song, gestures, parent=self,
                         muta_proba = self.muta_proba)

    def gen_sound(self, range_=None, fixed_normalize=True):
        """Generate the full song.
        
        if fixed_normalize = False, the song is normalized between -1 and 1
        """
        ab = self.gen_alphabeta(range_=range_, pad='last')
        out = synthesize(ab, fixed_normalize) # /!\ synthesize remove 2 samples
        assert np.isclose(np.nanmean(out), 0)
        if range_ is not None:
            expected_len = self.gesture_end(range_[-1]) - self.gestures[range_[0]][0]
        else:
            expected_len = len(self.song)
        assert len(out) == expected_len
        return out

    def gen_alphabeta(self, range_=None, pad=False):
        """Compute alpha and beta for the whole song."""
        if range_ is None:
            range_ = range(len(self.gestures))
        inner_pad = False
        length = self.gesture_end(range_[-1]) - self.gestures[range_[0]][0]
        if pad == 'last':
            # Add 2 samples, because the synthesizer shorten the signal by 2 samples
            length += 2
        elif pad:
            length += 2 * len(range_)
            inner_pad = True
        ab = np.zeros((length, 2))
        # true_start = When the first gesture starts
        true_start = self.gestures[range_[0]][0]
        for i in range_[:-1]:
            params = self.gestures[i][1]
            start = self.gestures[i][0] - true_start  # correct padding
            end = self.gesture_end(i) - true_start
            size = end - start
            if pad is True:
                # Add 2 samples, because the synthesizer shorten the signal by 2 samples
                end += 2
            assert size != 0
            ab[start:end] = gen_alphabeta(
                params, size,
                falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                falpha_nb_args=13, pad=inner_pad, beg=0)
        i = range_[-1]
        params = self.gestures[i][1]
        start = self.gestures[i][0] - true_start  # correct padding
        end = self.gesture_end(i) - true_start
        size = end - start
        # Add 2 samples, because the synthesizer shorten the signal by 2 samples
        if pad == 'last' or pad is True:
            end += 2
        ab[start:end] = gen_alphabeta(
            params, size,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13, pad=pad, beg=0)
        assert np.all(ab[:, 0] >= 0)
        return ab

    def gesture_end(self, i):
        """Return the end of a gesture."""
        if i < 0:
            i = len(self.gestures) + i # it works because i is negative
        try:
            end = self.gestures[i + 1][0]
        except IndexError:
            end = len(self.song)
        return end

    def shift_gesture(self, gesture, new_start):
        """Time shift the start of the gesture.
        Generate alpha beta parameters so that the new gesture shifted
        produces the same sound wave as the initial one at the time
        were the initial one was produced
        """
        g = gesture[1]
        start = gesture[0]
        t = new_start - start
        t /=bsa.SR
        ab_param = deepcopy(g)
        # -- Alpha --
        # new y-intercepts
        for i in [1, 5, 9]:
            ab_param[i] = g[i-1] * t + g[i] 
        # new phases
        for i in [2, 6, 10]:
            ab_param[i] = g[i] + 2 * np.pi * t * g[i+1]
        # -- Beta --
        # new y-intercept
        ab_param[14] = g[13] * t + g[14]
        # new phase
        ab_param[15] = g[15] + 2 * np.pi * t * g[16]

        return [new_start, ab_param]

    def mutate_test(self, n=1, to_move=None, new_pos=None):
        """Give a new song model with new GTEs.
        DEBUGGING function. Just to test the move mutation
        It is used by test_move_mutation.ipynb file
        to_move: choose the gesture to move
        new_pos: new start position of the gesture chosen
        n: number of mutations
        """
        gestures = deepcopy(self.gestures)
        for i in range(n):
            logger.info('moved')
            if to_move is None:
                to_move = self.rng.randint(1, len(gestures))
            else:
                to_move = to_move
            min_pos = gestures[to_move - 1][0] + 100
            print("min_pos =", min_pos)
            try:
                max_pos = gestures[to_move + 1][0] - 100
            except IndexError:  # Perhaps we have picked the last gesture
                max_pos = len(self.song) - 100
            print("max_pos =", max_pos)
            cur_pos = gestures[to_move][0]
            if cur_pos < min_pos or max_pos < cur_pos:  # not enough space
                logger.error('to_move < min_pos or max_pos < to_move')
            else:
                print("cur_pos =", cur_pos)
                if new_pos is None:
                    new_pos = self.rng.triangular(min_pos, cur_pos, max_pos)  # triangular distribution
                else:
                    new_pos = new_pos
                print("new_pos = ", new_pos)
                new_pos = int(new_pos)
                print("int(new_pos) = ", new_pos)
                if new_pos == cur_pos:
                    print("pas de changement")
                elif new_pos < cur_pos:
                    print("extension arriere")
                elif new_pos > cur_pos:
                    print("reduction en avant")
                else:
                    print("euuuh... ne dois jamais arriver")
                gestures[to_move] = self.shift_gesture(gestures[to_move], new_pos)
            # clean GTEs
            gestures.sort(key=lambda x: x[0])
            clean = False
            i_start = 1
            while not clean:
                for i in range(i_start, len(gestures)):
                    if gestures[i][0] - gestures[i - 1][0] < 100:
                        del gestures[i]
                        i_start = i
                        break
                else:  # If there is no break (for/else python syntax)
                    clean = True
            if len(self.song) - gestures[-1][0] < 100:
                del gestures[-1]
        return SongModel(self.song, gestures, parent=self,
                         muta_proba = self.muta_proba)