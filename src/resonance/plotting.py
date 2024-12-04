import string
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib_scalebar
import scipy.interpolate


def transect(pts, ppf, paui, duri, grid=True):
    if grid:
        duri,paui = np.meshgrid(duri,paui)
    transect = scipy.interpolate.griddata(pts, ppf.flatten(), (duri, paui), method='linear')
    transect = transect.flatten()
    return transect


def label_axes(fig=None, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if fig is None:
        fig = plt.gcf()

    if labels is None:
        labels = string.ascii_uppercase

    if "size" not in kwargs:
        kwargs["size"] = 24

    if 'weight' not in kwargs:
        kwargs['fontweight'] = 'heavy'

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (-0.2, 0.9)

    for ax in fig.axes:
        if 'colorbar' in ax.axes.get_label():
            continue
        lab = next(labels)
        ax.annotate(lab, xy=loc, xycoords="axes fraction", **kwargs)


def ppf(dur, pau, ppf, ax=None, colorbar=True):
    if ax is not None:
        plt.sca(ax)

    ax = plt.gca()
    plt.pcolor(dur, pau, ppf, cmap='Greys')
    plt.axis('square')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('Pulse [ms]')
    plt.ylabel('Pause [ms]')
    plt.xticks(np.arange(0, 20.1, 5))
    plt.yticks(np.arange(0, 20.1, 5))
    if colorbar:
        cax = inset_axes(plt.gca(), width="5%", height="20%", loc=1)
        cax = plt.colorbar(cax=cax, label='Phonotaxis', fraction=0.02, pad=0.03)
        cax.set_ticks([0, np.around(np.nanmax(ppf), decimals=2)])
    plt.sca(ax)


def pulse(x, c='k', alpha=0.5, dt=1, offset=0):
    T = np.arange(0, len(x)) / dt
    x[-1] = x[0]
    x += offset
    x[0] = offset
    x[-1] = offset
    plt.fill(T, x, c=c, alpha=alpha)
    plt.plot(T, x, c=c)
    plt.plot(T, np.zeros_like(x) + x[0], c=c)
    plt.axhline(offset, c='k', linewidth=0.5)


def despine(which="tr", axis=None):
    sides = {"t": "top", "b": "bottom", "l": "left", "r": "right"}

    if axis is None:
        axis = plt.gca()

    # Hide the spines
    for side in which:
        axis.spines[sides[side]].set_visible(False)

    # Hide the tick marks and labels
    if "r" in which:
        axis.yaxis.set_ticks_position("left")

    if "t" in which:
        axis.xaxis.set_ticks_position("bottom")

    if "l" in which:
        axis.yaxis.set_ticks([])

    if "b" in which:
        axis.xaxis.set_ticks([])


def scalebar(length, dx=1, units="", label=None, axis=None, location="lower right", frameon=False, **kwargs):
    """Add scalebar to axis.

    Usage:
        plt.subplot(122)
        plt.plot([0,1,2,3], [1,4,2,5])
        scalebar(0.5, 'femtoseconds', label='duration', location='lower right’)

    Args:
        length (float): Length of the scalebar in units of axis ticks - length of 1.0 corresponds to spacing between to major x-ticks
        dx (int, optional): Scale factor for length. E.g. if scale factor is 10, the scalebar of length 1.0 will span 10 ticks. Defaults to 1.
        units (str, optional): Unit label (e.g. 'milliseconds'). Defaults to ''.
        label (str, optional): Title for scale bar (e.g. 'Duration'). Defaults to None.
        axis (matplotlib.axes.Axes, optional): Axes to add scalebar to. Defaults to None (currently active axis - plt.gca()).
        location (str, optional): Where in the axes to put the scalebar (upper/lower/'', left/right/center). Defaults to 'lower right'.
        frameon (bool, optional): Add background (True) or not (False). Defaults to False.
        kwargs: location=None, pad=None, border_pad=None, sep=None,
                frameon=None, color=None, box_color=None, box_alpha=None,
                scale_loc=None, label_loc=None, font_properties=None,
                label_formatter=None, animated=False):

    Returns:
        Handle to scalebar object
    """

    if axis is None:
        axis = plt.gca()

    if "dimension" not in kwargs:
        kwargs["dimension"] = matplotlib_scalebar.dimension._Dimension(units)

    scalebar = ScaleBar(dx=dx, units=units, label=label, fixed_value=length, location=location, frameon=frameon, **kwargs)
    axis.add_artist(scalebar)
    return scalebar
