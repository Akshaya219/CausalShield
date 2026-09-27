import numpy as np
from sklearn.calibration import calibration_curve as _calibration_curve
from sklearn.isotonic import IsotonicRegression


class NumpyIsotonicRegression(IsotonicRegression):
    def transform(self, values):
        return np.asarray(super().transform(np.asarray(values)), dtype=float)


def calibration_curve(y_true, y_prob, n_bins=5, **kwargs):
    return _calibration_curve(y_true, y_prob, n_bins=n_bins, **kwargs)
