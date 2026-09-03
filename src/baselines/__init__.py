"""Point-estimate baselines for streaming learning.

Each keeps a single parameter vector (no particles) and exposes
predict_theta() / train(X, y) / observe_error(err) / n_resets.
"""

from .sgd import OnlineSGD
from .ph_sgd import PHSGD, PageHinkley
from .window_sgd import WindowSGD

__all__ = ["OnlineSGD", "PHSGD", "PageHinkley", "WindowSGD"]
