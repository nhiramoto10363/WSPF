"""Dataset loaders.

Each loader depends only on numpy, parses ARFF/CSV directly, and applies
leak-free preprocessing (PCA or standardization fit on an initial segment).
"""

from .email_loader import EmailDataLoader
from .insects_loader import InsectsDataLoader

__all__ = [
    "EmailDataLoader",
    "InsectsDataLoader",
]
