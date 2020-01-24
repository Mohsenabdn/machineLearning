"""
This file includes some auxiliary functions needed for predictive models or for
creating random data sets.
"""
import numpy as np


def Donut(n, r, margin):
    """
    This function creates two group of random points. The first group form a
    donut and the other group form the hole in the center of the donut.

    :param n: The number of all random points.
    :param r: Value of shrinkage or expansion of the random points.
    :param margin: A margin distance between hole and donut.
    :return: Two numpy array x and y. x consists of positions of donut and hole
        points and y is their colors.
    """
    x = np.random.randn(n, 2)
    x_donut = x[np.sqrt(np.sum(x ** 2, axis=1)) > 1] * (r + margin / 2)
    x_hole = x[np.sqrt(np.sum(x ** 2, axis=1)) <= 1] * (r - margin / 2)

    y_hole = np.zeros([x_hole.shape[0], 1])
    y_donut = np.ones([x_donut.shape[0], 1])

    x = np.vstack([x_hole, x_donut])
    y = np.vstack([y_hole, y_donut])
    return x, y


def Sigmoid(z):
    """
    This function creates a bell-shaped curve ranged between 0 and 1. It is
    used in binary logistic regression model to map the regression line, plane
    or hyperplane into (0,1).

    :param z: A numpy ndarray.
    :return: A numpy ndarray with values in (0,1).
    """
    return 1 / (1+np.exp(-z))


def SoftMax(z):
    """
    This function maps the regression line, plane, or hyperplane obtained by
    logistic regression model, into (0,1). It is used when the target is not
    binary (see Sigmoid function).

    :param z: A numpy ndarray.
    :return: A numpy ndarray with values in (0,1). the sum of each row is 1 when
    there is only one target column.
    """
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
