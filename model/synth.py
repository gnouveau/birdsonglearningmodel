"""Synthesizer module for bird songs.

This module relies on the C program located at
../../csynthesizer/alphabeta2dat. It call this program with the alpha and
beta given.
"""

import numpy as np
import birdsynth
import birdsonganalysis as bsa


def exp_sin(x, p, nb_exp=2, nb_sin=2):
    """Generator function for parameters with a mixture of exp and sin."""
    ip = np.nditer(p)
    return np.sum([next(ip) * np.exp(-np.abs(next(ip)) * x)
                   for i in range(nb_exp)]
                  + [next(ip) * np.sin(next(ip) + (2 * np.pi * x) * next(ip))
                     for i in range(nb_sin)] + [next(ip)], axis=0)


def only_sin(x, p, nb_sin=3):
    """Generator function for parameters with only sinuses."""
    ip = np.nditer(p)
    return np.sum([(next(ip) * x + next(ip))
                   * np.sin(next(ip) + (2*np.pi * x) * next(ip))
                   for i in range(nb_sin)] + [next(ip)], axis=0)


def exp_sin_str(p, nb_exp=2, nb_sin=2, x='x'):
    """Return string representation of `exp_sin`."""
    ip = np.nditer(p)
    out = ''
    for i in range(nb_exp):
        out += '{:.2} exp({:.2} {}) + '.format(float(next(ip)),
                                               -np.abs(float(next(ip))), x)
    for i in range(nb_sin):
        out += '{:.2f} sin(({:.2f} + 2π {})/{:.2f}) + '.format(
            float(next(ip)), float(next(ip)), x, float(next(ip)))
    out += '{:.2f}'.format(float(next(ip)))
    return out


def gen_sound(params, length, falpha, fbeta, falpha_nb_args, beg=0):
    """Generate a sound with parameters and alpha and beta generators.

    params - The parameters for falpha and fbeta, concatenated
    length - The length of the alpha_beta 2D array to generate
    falpha - The generator function for alpha of signature
             falpha(t:np.array[t], params:np.array[nb_alpha_params]):
                np.array[t]
    fbeta - The genenator function for beta of signature
             fbeta(t:pn.array[t], params:np.array[nb_beta_params]):
                np.array[t]
    falpha_nb_args - Number of params falpha needs. It will be used for
                     the slicing of `params`. Indeed, in the code, we do
    beg - The value to begin with, in sample

    ```
    falpha(t, params[:falpha_nb_args])
    fbeta(t, params[falpha_nb_args:])
    ```

    Returns - 1D numpy.array with the normalized signal between -1 and 1
    """
    alpha_beta = gen_alphabeta(params, length, falpha, fbeta, falpha_nb_args,
                               pad=True, beg=beg)
    out = synthesize(alpha_beta)
    assert len(out) == length
    return out


def gen_alphabeta(params, length, falpha, fbeta,
                  falpha_nb_args, beg, pad=True):
    """Generate a alpha_beta 2D array.

    params - The parameters for falpha and fbeta, concatenated
    length - The length of the alpha_beta 2D array to generate
    falpha - The generator function for alpha of signature
             falpha(t:np.array[t], params:np.array[nb_alpha_params]):
                np.array[t]
    fbeta - The genenator function for beta of signature
             fbeta(t:np.array[t], params:np.array[nb_beta_params]):
                np.array[t]
    falpha_nb_args - Number of params falpha needs. It will be used for
                     the slicing of `params`. Indeed, in the code, we do
    pad - Should we add the padding to correct the csynth bug or not #TODO: quel bug ?

    ```
    falpha(t, params[:falpha_nb_args])
    fbeta(t, params[falpha_nb_args:])
    ```

    Returns - A 2D numpy.array of shape (length, 2) with in the first column
    the alpha parameters and in the second the beta parameters.
    """
    if pad: # TODO: demander c'est quoi ce padding ?
        padding = 2
    else:
        padding = 0
    # + 2 padding is necessary with ba synth.
    # Add a float to all the values of the numpy ndarray
    t = beg / bsa.SR + np.linspace(0, (length+2)/bsa.SR, length + padding)
    alpha_beta = np.stack(
        (
            falpha(t, params[:falpha_nb_args]),
            fbeta(t, params[falpha_nb_args:])
        ), axis=-1)
        
    alpha_beta[:, 0] = np.where(alpha_beta[:, 0] < 0, 0, alpha_beta[:, 0])
    # TODO: Try beta with only negative values
    alpha_beta[:, 1] = np.where(alpha_beta[:, 1] > 0, 0, alpha_beta[:, 1])
    
    return alpha_beta


def synthesize(alpha_beta, fixed_normalize=False, boundary=150000):
    """Return the song signal given the alpha beta parameters.

    This function reduce by 2 samples the song produced

    alpha_beta - A 2d numpy.array of shape (length, 2)
                 with alpha on the alpha_beta[:, 0] elements
                 and beta on the alpha_beta[:, 1] elements

    Returns - 1D numpy.array with the normalized signal between -1 and 1
    """
    out = birdsynth.synth(alpha_beta)
    assert not np.any(np.isnan(out))
    if fixed_normalize:
        scale = boundary
    else:
        scale = np.nanmax(out) - np.nanmin(out)
    out = 2 * (out - np.nanmin(out)) / scale - 1
    out = out - np.nanmean(out)
    return out
