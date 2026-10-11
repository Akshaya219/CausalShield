import numpy as np
from sklearn.isotonic import IsotonicRegression


class NumpyIsotonicRegression(IsotonicRegression):
    def transform(self, values):
        return np.asarray(super().transform(np.asarray(values)), dtype=float)

