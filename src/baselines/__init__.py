"""データストリーム学習ベースライン (R2-4)。

いずれも粒子を持たない点推定器で、共通インタフェース
    predict_theta() / train(X, y) / observe_error(err) / n_resets
を提供する。
"""

from .sgd import OnlineSGD
from .ph_sgd import PHSGD, PageHinkley
from .window_sgd import WindowSGD

__all__ = ["OnlineSGD", "PHSGD", "PageHinkley", "WindowSGD"]
