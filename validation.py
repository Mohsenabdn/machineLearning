import numpy as np


def CrossValidation(k, x, y, model, metric_func):
    """
    This function do k-fold crossvalidation on the selected model.

    :param k: Number of folds (integer).
    :param x: A ndarray as the features.
    :param y: A ndarray as the target.
    :param model: The selected model.
    :param metric_func: A function to evaluate the performance of the model."""

    idx = np.random.permutation(x, shape[0])
    x = x[idx, :]
    y = y[idx, :]

    folds_x = []
    folds_y = []
    for i in range(k):
        folds_x.append(x[i::k])
        folds_y.append(y[i::k])

    for i in range(k):
        val_x = folds_x[i]
        val_y = folds_y[i]

        train_x = np.vstack(folds_x[:i] + folds_x[(i+1):])
        train_y = np.vstack(folds_y[:i] + folds_y[(i+1):])

        model.fit(train_x, train_y)
        y_hat = model.predict(val_x)
        performance.append(metric_func(val_y, y_hat))

    return np.mean(performancece)
