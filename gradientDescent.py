
import numpy as np


class GradientDescent:

    def __init__(self, num_iter=1000, tol=1e-4, learning_rate=1e-4):
        self.num_iter = num_iter
        self.tolerance = tol
        self.learning_rate = learning_rate
        self.beta = 0

    def apply(self, x, y, pad=False):
        """
        This method get x and y nd arrays and apply the gradient descent method.

        :param x: nd array
        :param y: nd array
        :param pad: boolean arguement to add y-intercept

        :return: y_hat
        """
        loss = [1000]

        if pad:
            x = np.hstack([np.ones((x.shape[0], 1)), x])

        self.beta = np.random.randn(x.shape[1], y.shape[1])

        for i in range(self.num_iter):
            y_hat = x @ self.beta
            loss.append(np.trace((y - y_hat).T @ (y - y_hat)/y.shape[0]))
            grad = -(y - y_hat).T @ x
            self.beta -= self.learning_rate * grad.T
            if abs(loss[-1] - loss[-2]) < self.tolerance:
                break
        return y_hat
