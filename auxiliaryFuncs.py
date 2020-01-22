"""
This file includes some auxiliary functions needed for predictive models.
"""
import numpy as np


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
