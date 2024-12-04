import numpy as np
import jax.numpy as jnp
import scipy.signal.windows
from itertools import product
import jax.scipy as jsc
from jax import jit
from functools import partial
from scipy.optimize import minimize
import matplotlib.pyplot as plt


@jit
def conv(x, filter):
    """filter...

    Args:
        x ([type]): [description]
        filter (np.array like)

    Returns:
        [type]: [description]
    """
    if filter.ndim==1 and x.ndim==2:
        filter = filter[:, np.newaxis]
    x = jsc.signal.convolve2d(x, filter[:x.shape[0]-1,:], mode='full', method='fft')
    return x

@jit
def conv1d(x, filter):
    """filter...

    Args:
        x ([type]): [description]
        filter (np.array like)

    Returns:
        [type]: [description]
    # """
    x = jsc.signal.convolve(x, filter, mode='full')
    return x


# @partial(jit, static_argnums=(0,))
def gausswin(N: float, a: float = 2.5):
    N = N-1
    if N<=1:
        w = jnp.ones((1,))
    else:
        n = jnp.arange(-N/2, N/2)
        w = jnp.exp(-(1/2)*(a*n/(N/2))**2)
    return w


def makePPFstim(ppauMax=80, pdurMax=80, ppauMin=0, pdurMin=0, cdur=200, cpau=200, Fs=1000, step=1):
    """_summary_

    Args:
        ppauMax (int, optional): _description_. Defaults to 80 ms.
        pdurMax (int, optional): _description_. Defaults to 80 ms.
        ppauMin (int, optional): _description_. Defaults to 0 ms.
        pdurMin (int, optional): _description_. Defaults to 0 ms.
        cdur (int, optional): _description_. Defaults to 200 ms.
        cpau (int, optional): _description_. Defaults to 200 ms.
        Fs (int, optional): _description_. Defaults to 1000 Hz.
        step (int, optional): _description_. Defaults to 1  ms.

    Returns:
        _type_: _description_
    """
    dwnSmp = 1000 / Fs
    ppau = (np.arange(ppauMin, ppauMax, step) / dwnSmp).astype(np.uint64)
    pdur = (np.arange(pdurMin, pdurMax, step) / dwnSmp).astype(np.uint64)
    cpau = int(cpau / dwnSmp)
    cdur = int(cdur / dwnSmp)

    stim_dict = dict()
    stim_dict['ppau'] = np.zeros((len(ppau)*len(pdur),), dtype=np.uint64)
    stim_dict['pdur'] = np.zeros((len(ppau)*len(pdur),), dtype=np.uint64)
    stim_dict['npul'] = np.zeros((len(ppau)*len(pdur),), dtype=np.uint64)
    stim_dict['stim'] = []
    for cnt, (pau, dur) in enumerate(product(ppau, pdur)):
        pulse = np.concatenate([np.ones((dur, 1)), np.zeros((pau, 1))], axis=0)
        stim_dict['npul'][cnt] = max(1, int(np.floor(cdur / (len(pulse) + 0.000001))))
        chirp = np.tile(pulse[:, 0], reps=int(stim_dict['npul'][cnt]))[:, np.newaxis]
        stim_dict['stim'].append(np.concatenate([np.zeros((50, 1)), chirp, np.zeros((cdur + cpau - len(chirp), 1))]))
        stim_dict['ppau'][cnt] = pau
        stim_dict['pdur'][cnt] = dur

    stim_dict['pper'] = stim_dict['pdur'] + stim_dict['ppau']
    stim_dict['pdc'] = stim_dict['pdur'] / (stim_dict['pper'] + 0.000001)
    stim_dict['cpau'] = cpau * np.ones_like(stim_dict['ppau'])
    stim_dict['cdur'] = cdur * np.ones_like(stim_dict['ppau'])
    stim_dict['cper'] = stim_dict['pdur'] + stim_dict['ppau']
    stim_dict['cdc'] = stim_dict['pdur'] / (stim_dict['pper'] + 0.000001)

    stim_matrix = jnp.concatenate(stim_dict['stim'], axis=1)
    return stim_matrix, stim_dict, ppau, pdur


def makePPFstim_geom(ppauMax=80, pdurMax=80, ppauMin=None, pdurMin=None, cdur=200, cpau=200, Fs=1000, num=16, include_zero: bool = True):
    """_summary_

    Args:
        ppauMax (int, optional): _description_. Defaults to 80 ms.
        pdurMax (int, optional): _description_. Defaults to 80 ms.
        ppauMin (_type_, optional): _description_. Defaults to 1/Fs ms.
        pdurMin (_type_, optional): _description_. Defaults to 1/Fs.
        cdur (int, optional): _description_. Defaults to 200 ms.
        cpau (int, optional): _description_. Defaults to 200 ms.
        Fs (int, optional): _description_. Defaults to 1000 Hz.
        num (int, optional): Number of pau and dur values. Total number of stims will be num^2. Defaults to 16.
        include_zero (bool, optional): Prepend zero pau and dur. Defaults to True.

    Returns:
        _type_: _description_
    """
    dwnSmp = 1000 / Fs
    if ppauMin is None:
        ppauMin = 1000 / Fs

    if pdurMin is None:
        pdurMin = 1000 / Fs

    if include_zero:
        num = num - 1

    ppau = (np.geomspace(ppauMin, ppauMax, num) / dwnSmp).astype(np.uint64)
    pdur = (np.geomspace(pdurMin, pdurMax, num) / dwnSmp).astype(np.uint64)
    if include_zero:
        pdur = np.insert(pdur, 0, 0)
        ppau = np.insert(ppau, 0, 0)
    cpau = int(cpau / dwnSmp)
    cdur = int(cdur / dwnSmp)

    stim_dict = dict()
    stim_dict['ppau'] = np.zeros((len(ppau)*len(pdur),), dtype=np.uint64)
    stim_dict['pdur'] = np.zeros((len(ppau)*len(pdur),), dtype=np.uint64)
    stim_dict['npul'] = np.zeros((len(ppau)*len(pdur),), dtype=np.uint64)
    stim_dict['stim'] = []
    for cnt, (pau, dur) in enumerate(product(ppau, pdur)):
        pulse = np.concatenate([np.ones((dur, 1)), np.zeros((pau, 1))], axis=0)
        stim_dict['npul'][cnt] = max(1, int(np.floor(cdur / (len(pulse) + 0.000001))))
        chirp = np.tile(pulse[:, 0], reps=int(stim_dict['npul'][cnt]))[:, np.newaxis]
        stim_dict['stim'].append(np.concatenate([np.zeros((50, 1)), chirp, np.zeros((cdur + cpau - len(chirp), 1))]))
        stim_dict['ppau'][cnt] = pau
        stim_dict['pdur'][cnt] = dur

    stim_dict['pper'] = stim_dict['pdur'] + stim_dict['ppau']
    stim_dict['pdc'] = stim_dict['pdur'] / (stim_dict['pper'] + 0.000001)
    stim_dict['cpau'] = cpau * np.ones_like(stim_dict['ppau'])
    stim_dict['cdur'] = cdur * np.ones_like(stim_dict['ppau'])
    stim_dict['cper'] = stim_dict['pdur'] + stim_dict['ppau']
    stim_dict['cdc'] = stim_dict['pdur'] / (stim_dict['pper'] + 0.000001)

    stim_matrix = jnp.concatenate(stim_dict['stim'], axis=1)
    return stim_matrix, stim_dict, ppau, pdur


def showPPF(model_output, neuron="LN4"):
    ppf_data = jnp.nanmean(model_output[neuron], axis = 0)
    ppf_shape = int(np.sqrt(ppf_data.shape[0]))

    # Duration is already x and pause is y
    ax = plt.imshow(ppf_data.reshape(ppf_shape, ppf_shape), origin="lower", cmap="Greens")
    plt.xlabel("Duration[ms]")
    plt.ylabel("Pause [ms]")
    plt.xticks([0, 19, 39], [0, 40, 80])
    plt.yticks([0, 19, 39], [0, 40, 80])
    plt.colorbar()

    return ax