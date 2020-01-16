import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self):
        self.betas = 0
        self.beta_0 = 0

    def fit(self, x, y, pad=False):
        if pad:
            pad = np.ones(x.shape[0])
            x = np.concatenate(pad, x, axis=1)
        self.betas = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

    def predict(self, x, pad=False):
        if pad:
            pad = np.ones(x.shape[0])
            x = np.concatenate(pad, x, axis=1)
        return np.dot(x, self.betas)

# Scoring function


def R2(y1, y2):
    """ Computing R^2 value for predicted targets and actual targets.

    Inputs:
    y1 : Actual target values (An array with shape (m,1) where m is a positive
    integer).
    y2 : predicted target values with the same shape as y1's.

    Output:
    A real value between 0 and 1. The greater the value, the more exact
    prediction."""
    yBar = np.mean(y1)
    top = np.sum((y1-y2)**2)
    bot = np.sum((y1-yBar)**2)
    return 1 - top/bot


# Initializing random features, response, and parameters.
n = 1000
x0 = np.ones((n, 1))
x1 = np.random.rand(n, 1)*10 - 5
x2 = np.random.rand(n, 1)*6 - 3
x3 = np.random.rand(n, 1)*14 - 7
x = np.concatenate((x0, x1, x2, x3), axis=1)
betas = np.array([[1], [2], [3], [4]])
y = np.dot(x, betas) + np.random.rand(n).reshape((n, 1))*10

# instansiating, fitting, predicting and scoring
lr = LinearRegression()
lr.fit(x, y)
yPred = lr.predict(x)
print(R2(y, yPred))
