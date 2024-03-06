import numpy as np


class QDA:
    def __init__(self) -> None:
        self.means = []
        self.covariances = [0, 0]
        self.priors = []

    def fit(self, X, y):
        binary_values, counts = np.unique(y, return_counts=True)
        n = len(y)
        self.priors = counts / n
        self.means = np.array(
            [X[y == zero_or_one].mean(axis=0) for zero_or_one in binary_values]
        )
        self.covariances[0] = np.cov(X[y == 0], rowvar=False)
        self.covariances[1] = np.cov(X[y == 1], rowvar=False)

    def _feature_probability(self, x_test, mean, cov):

        k = x_test.shape[1]
        first_part = 1 / np.sqrt((2 * np.pi) ** k * np.linalg.det(cov))
        diff = x_test - mean

        second_part = np.exp(
            -0.5
            * np.einsum(
                "ij,ij->i",
                np.matmul((diff), np.linalg.inv(cov)),
                diff,
            ),
        )
        return first_part * second_part

    def predict(self, Xtest):
        prob = self.predict_proba(Xtest)
        return (prob > 0.5).astype(int)

    def predict_proba(self, Xtest):
        return (
            self._feature_probability(Xtest, self.means[1], self.covariances[1])
            * self.priors[1]
            / (
                self._feature_probability(Xtest, self.means[0], self.covariances[0])
                * self.priors[0]
                + self._feature_probability(Xtest, self.means[1], self.covariances[1])
                * self.priors[1]
            )
        )

    def get_params(self):
        return (self.means, self.covariances, self.priors)
