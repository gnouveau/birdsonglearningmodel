"""Define the SongModel class."""

from copy import deepcopy

import numpy as np
import logging
from synth import only_sin, gen_alphabeta, synthesize


logger = logging.getLogger('songmodel')


class SongModel:
    """Song model structure."""

    def __init__(self, song, gestures=None, nb_split=20, rng=None,
                 parent=None, priors=None):
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

    def mutate(self, n=1):
        """Give a new song model with new GTEs.
        n: number of mutations
        """
        gestures = deepcopy(self.gestures)
        for i in range(n):
            act = self.rng.uniform()
            if act < 0.2 and len(gestures) > 2:  # Delete a gesture
                logger.info('deleted')
                to_del = self.rng.randint(len(gestures))
                del gestures[to_del]
                # if the first gesture is suppressed,
                # the new one has to start at 0
                if to_del == 0:
                    gestures[0][0] = 0
            elif act < 0.4:  # Add a new gesture
                logger.info('added')
                add_after = self.rng.randint(len(gestures) - 1)
                try:
                    add_at = self.rng.randint(gestures[add_after][0] + 100,
                                           gestures[add_after + 1][0] - 100)
                except ValueError:  # There is no new place
                    continue
                gestures.insert(add_after + 1,
                                [add_at, deepcopy(gestures[add_after][1])])
            elif act < 0.6:  # Take a gesture and put it in another gesture
                logger.info('copied')
                from_, dest = self.rng.randint(len(gestures), size=2)
                gestures[dest][1] = deepcopy(gestures[from_][1])
            elif act < 0.8:  # Move where the gesture start
                logger.info('moved')
                to_move = self.rng.randint(1, len(gestures))
                min_pos = gestures[to_move - 1][0] + 100
                try:
                    max_pos = gestures[to_move + 1][0] - 100
                except IndexError:  # Perhaps we have picked the last gesture
                    logger.debug('last gesture picked')
                    max_pos = len(self.song) - 100
                new_pos = self.rng.normal(loc=gestures[to_move][0],
                                          scale=max((max_pos-min_pos)/4,
                                                    0.00005))
                gestures[to_move][0] = int(np.clip(new_pos, min_pos, max_pos))
            else:  # Do not mutate
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
        return SongModel(self.song, gestures, parent=self)

    def gen_sound(self, range_=None, fixed_normalize=True):
        """Generate the full song."""
        ab = self.gen_alphabeta(range_=range_, pad='last')
        out = synthesize(ab, fixed_normalize)
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
        if pad == 'last': # TODO: pkoi ajouter 2 valeurs a la fin ?
            length += 2
        elif pad: # TODO: quel est l'interet de pad ?
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
            if pad is True: # TODO: pkoi ajouter 2 valeurs ?
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
        if pad == 'last' or pad is True: # TODO: pkoi ajouter 2 valeurs a la fin ?
            end += 2
        ab[start:end] = gen_alphabeta(
            params, size,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13, pad=pad, beg=0)
        assert np.all(ab[:, 0] >= 0) # TODO: pkoi checker ca ?
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
