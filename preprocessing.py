
import numpy as np


def scale(x, interval=[0, 1], cols: list = []):
    """
    This function scales selected columns of a nd array to the given range.

    :param x: nd array
    :param interval: the range for scaling
    :param cols: columns to be scaled

    :return: x after scaling selected columns."""

    if not cols:
        cols = list(range(x.shape[1]))
    top = x[:, cols] - np.min(x[:, cols], axis=0)
    bot = np.max(x[:, cols], axis=0) - np.min(x[:, cols], axis=0)
    x[:, cols] = top/bot
    x[:, cols] = (1 - x[:, cols])*interval[0] + x[:, cols]*interval[1]
    return x


def gaussianScaling(x, sample_mean_std=True, cols: list = []):
    """
    This function scales selected columns of a nd array to normal distribution.

    :param x: nd array
    :param sample_mean_std: boolean type. if True, the function uses mean and
     standard deviation of the sample.
    :param cols: columns to be scaled.

    :return: x after scaling selected columns."""

    if not cols:
        cols = list(range(x.shape[1]))
    if sample_mean_std:
        mu = np.mean(x[:, cols], axis=0)
        sigma = np.std(x[:, cols], axis=0)
    else:
        mu = 0
        sigma = 1
    x[:, cols] -= mu
    x[:, cols] /= sigma
    return x


def robustScale(x, cols: list = []):
    """
    This function scales selected columns of a nd array to using first and 3rd
     percentile (the scaled values are not necessarily betwee 0 and 1)

    :param x: nd array
    :param cols: columns to be scaled

    :return: x after scaling selected columns."""
    if not cols:
        cols = list(range(x.shape[1]))
    top = x[:, cols] - np.percentile(x[:, cols], 25, axis=0)
    bot = np.percentile(x[:, cols], 75, axis=0) - np.percentile(x[:, cols], 25, axis=0)
    x[:, cols] = top / bot
    return x
