from numba import jit, prange
import numpy as np


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def raf(params, stim, fs: float = 1000.0):

    Vr = 1
    x = 0
    y = 0
    dt = 1 / fs

    stim_amp = params[0] / dt
    b = params[1]
    w = 109 * 2 * np.pi#params[2] * 2 * np.pi
    nonlinearity_gain = params[3] / dt
    tx = 1#params[4]
    ty = 1#params[5]

    VV = np.zeros((*stim.shape, 4))

    for t in range(len(stim)):
        x = x + dt * tx * (b * x - w * y + stim_amp * stim[t])  # current
        y = y + dt * ty * (w * x + b * y)  # voltage

        VV[t, 3] = stim[t]
        VV[t, 0] = y
        if y >= Vr:
            VV[t, 2] = nonlinearity_gain
            y = Vr
            x = 0

        VV[t, 1] = x
    return VV


@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=True)
def run_model(stims, params, fs):
    outs = np.empty((stims.shape[1], stims.shape[0], 4))
    for cnt in prange(stims.shape[1]):
        outs[cnt, ...] = raf(params, stims[:, cnt], fs)

    return outs.T
