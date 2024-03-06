import numpy as np


class LDA:
    def __init__(self):
        self.means = []
        self.covariance = None
        self.priors = []

    def _feature_probability(self, x_test, mean):

        k = x_test.shape[1]
        first_part = 1 / np.sqrt((2 * np.pi) ** k * np.linalg.det(self.covariance))
        diff = x_test - mean

        second_part = np.exp(
            -0.5
            * np.einsum(
                "ij,ij->i",
                np.matmul((diff), np.linalg.inv(self.covariance)),
                diff,
            ),
        )
        return first_part * second_part

    def fit(self, X, y):
        binary_values, counts = np.unique(y, return_counts=True)
        n = len(y)
        self.priors = counts / n
        self.means = np.array(
            [X[y == zero_or_one].mean(axis=0) for zero_or_one in binary_values]
        )
        cov1 = np.cov(X[y == 1], rowvar=False)
        cov0 = np.cov(X[y == 0], rowvar=False)
        n1 = sum(y)
        n0 = len(y) - n1
        self.covariance = ((n1 - 1) * cov1 + (n0 - 1) * cov0) / (n1 + n0 - 2)

    def predict(self, Xtest):
        prob = self.predict_proba(Xtest)
        return (prob > 0.5).astype(int)

    def get_line(self):
        delta_mu = self.means[1] - self.means[0]
        A = delta_mu[0]
        B = delta_mu[1]
        M = (self.means[1] + self.means[0]) / 2
        C = -A * M[0] - B * M[1]
        x1, x2 = -2, 4
        y1 = (-C - A * x1) / B
        y2 = (-C - A * x2) / B
        return x1, x2, y1, y2

    def predict_proba(self, Xtest):
        return (
            self._feature_probability(Xtest, self.means[1])
            * self.priors[1]
            / (
                self._feature_probability(Xtest, self.means[0]) * self.priors[0]
                + self._feature_probability(Xtest, self.means[1]) * self.priors[1]
            )
        )

    def get_params(self):
        return (self.means, self.covariance, self.priors)
