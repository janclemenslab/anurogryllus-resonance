"""
see kernel.m
"""
import numpy as np
from numba import jit


@jit # @partial(jit, static_argnums=(0,))
def exponential(duration, tau):
    """ kernel = exponential(duration, tau)
        normalized exponential kernel: EXP(-t/tau), width TAU in points."""
    if duration>=1:
        kernel = np.exp(-(np.arange(0, duration-1)/tau))/tau
    else:
        kernel = np.ones((1,))
    return kernel
