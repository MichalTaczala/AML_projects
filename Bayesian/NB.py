import numpy as np


class NB:
    def __init__(self):
        self.means = []
        self.variances = [None, None]
        self.priors = []

    def fit(self, X, y):
        binary_values, counts = np.unique(y, return_counts=True)
        n = len(y)
        self.priors = counts / n

        self.means = np.array(
            [X[y == zero_or_one].mean(axis=0) for zero_or_one in binary_values]
        )
        self.variances[0] = np.var(X[y == 0], axis=0)
        self.variances[1] = np.var(X[y == 1], axis=0)

    def predict(self, Xtest):
        prob = self.predict_proba(Xtest)
        return (prob > 0.5).astype(int)

    def predict_proba(self, Xtest):
        p = Xtest.shape[1]
        prob_y1 = np.prod(
            np.sqrt(1 / (2 * np.pi) / self.variances[1])
            * np.exp(-(((Xtest - self.means[1]) / self.variances[1]) ** 2) / 2),
            axis=1,
        )

        prob_y0 = np.prod(
            np.sqrt(1 / (2 * np.pi) / self.variances[0])
            * np.exp(-(((Xtest - self.means[0]) / self.variances[0]) ** 2) / 2),
            axis=1,
        )

        total_probability = (
            self.priors[1]
            * prob_y1
            / (self.priors[0] * prob_y0 + self.priors[1] * prob_y1)
        )
        return total_probability

    def get_params(self):
        return self.means, self.variances, self.priors
