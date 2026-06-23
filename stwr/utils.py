"""
Visualization utilities for GWR and MGWR parameter surfaces.

Requires matplotlib and geopandas.
"""

import numpy as np
from libpysal.common import requires


@requires('matplotlib')
def shift_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """
    Shift the center of a colormap so that ``midpoint`` maps to the data
    value 0. Useful for diverging data with negative and positive values.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The colormap to shift.
    start : float (0.0 to midpoint)
        Offset from the low end. Default 0.0.
    midpoint : float (0.0 to 1.0)
        New center of the colormap. Should typically be
        1 - vmax / (vmax + abs(vmin)). Default 0.5 (no shift).
    stop : float (midpoint to 1.0)
        Offset from the high end. Default 1.0.
    name : str
        Name for the new colormap.

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap
    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    new_cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=new_cmap)

    return new_cmap


@requires('matplotlib')
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncate a colormap to use only a sub-range of its full spectrum.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The colormap to truncate.
    minval : float
        Lower end of the retained sub-range (0.0 to 1.0).
    maxval : float
        Upper end of the retained sub-range (0.0 to 1.0).
    n : int
        Number of color gradation steps.

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap
    """

    import matplotlib as mpl

    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


@requires('matplotlib')
@requires('geopandas')
def compare_surfaces(data, var1, var2, gwr_t, gwr_bw, mgwr_t, mgwr_bw, name,
                     kwargs1, kwargs2, savefig=None):
    """
    Create side-by-side comparison maps of GWR and MGWR parameter surfaces.

    Non-significant areas (t-value = 0) are shown in light grey. The
    colorbar is centered at zero for diverging data.

    Parameters
    ----------
    data : geopandas.GeoDataFrame
        DataFrame containing the GWR and MGWR results.
    var1 : str
        Column name for the GWR parameter estimates.
    var2 : str
        Column name for the MGWR parameter estimates.
    gwr_t : str
        Column name for GWR t-values (associated with var1).
    gwr_bw : float
        GWR bandwidth for this variable.
    mgwr_t : str
        Column name for MGWR t-values (associated with var2).
    mgwr_bw : float
        MGWR bandwidth for this variable.
    name : str
        Common variable name for the plot title.
    kwargs1 : dict
        Additional keyword arguments for the GWR surface plot.
    kwargs2 : dict
        Additional keyword arguments for the MGWR surface plot.
    savefig : str, optional
        Path to save the figure. Default None (display only).
    """
    import matplotlib.pyplot as plt
    import geopandas as gp

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45, 20))
    ax0 = axes[0]
    ax0.set_title('GWR ' + name + ' Surface (BW: ' + str(gwr_bw) + ')',
                  fontsize=40)
    ax1 = axes[1]
    ax1.set_title('MGWR ' + name + ' Surface (BW: ' + str(mgwr_bw) + ')',
                  fontsize=40)

    #Set color map
    cmap = plt.cm.seismic

    #Find min and max values of the two combined datasets
    gwr_min = data[var1].min()
    gwr_max = data[var1].max()
    mgwr_min = data[var2].min()
    mgwr_max = data[var2].max()
    vmin = np.min([gwr_min, mgwr_min])
    vmax = np.max([gwr_max, mgwr_max])

    #If all values are negative use the negative half of the colormap
    if (vmin < 0) & (vmax < 0):
        cmap = truncate_colormap(cmap, 0.0, 0.5)
    #If all values are positive use the positive half of the colormap
    elif (vmin > 0) & (vmax > 0):
        cmap = truncate_colormap(cmap, 0.5, 1.0)
    #Otherwise, there are positive and negative values so the colormap so zero is the midpoint
    else:
        cmap = shift_colormap(cmap, start=0.0,
                              midpoint=1 - vmax / (vmax + abs(vmin)), stop=1.)

    #Create scalar mappable for colorbar and stretch colormap across range of data values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=vmin, vmax=vmax))

    #Plot GWR parameters
    data.plot(var1, cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **kwargs1)
    if (gwr_t == 0).any():
        data[gwr_t == 0].plot(color='lightgrey', ax=ax0, **kwargs2)

    #Plot MGWR parameters
    data.plot(var2, cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **kwargs1)
    if (mgwr_t == 0).any():
        data[mgwr_t == 0].plot(color='lightgrey', ax=ax1, **kwargs2)

    #Set figure options and plot
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=50)
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()