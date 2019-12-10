# This file includes some useful functions.
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def plotFunc(x, y, mark=('.', 8), c='b', grid=True, yformat='{x:,.4f}',
             rot='horizontal', **kwargs):
    """ This function plots a line or curve in a plane.

        Parameters
        ----------
        x : A list-like object for horizontal axis.
        y : A list-like object for vertical axis.
        mark : A tuple including the type of marker (first argument) and the
            size of marker (the second argument) (default ('.', 8)).
        c : A string object showing the color of plotted line or curve
            (default 'b').
        grid : A boolean object whether plotting grid or not (default True).
        yformat : A string object representing the format of y ticks (default
            '{x:,.4f}'). The default value indicates 4 decimals for y ticks.
        rot : 'horizontal', 'vertical' or any number representing the rotation
            of x ticks.

        Other Parameters
        ----------------
        **kwargs : including strings representing title, xlabel, and ylabel.
            Also, it includes xlim and ylim as tuples and xticks and yticks as
            list-like objects.

        Return
        ------
        A 2D graph."""

    plt.plot(x, y, color=c, linestyle='-', marker=mark[0], markersize=mark[1])
    plt.gca().set(**kwargs)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter(yformat))
    if 'xticks' in kwargs.keys():
        plt.gca().set_xticklabels([str(i) for i in kwargs['xticks']],
                                  rotation=rot)
    else:
        plt.xticks(rotation=rot)
    if grid:
        plt.grid(linestyle='--')
