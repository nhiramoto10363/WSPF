"""Neural network models.

  regression_nn      Gaussian-likelihood regression MLP
  binary_nn          binary classification MLP (email)
  classification_nn  multiclass classification MLP (INSECTS)
"""

from .regression_nn import (
    NeuralNetRegression,
    create_regression_grad_fn,
    create_regression_loglik_fn,
    create_regression_per_sample_grad_fn,
)
from .binary_nn import (
    NeuralNetModel,
    create_nn_grad_fn,
    create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
    generate_nn_stream_data,
)
from .classification_nn import (
    MulticlassNeuralNetModel,
    create_mc_grad_fn,
    create_mc_loglik_fn,
    create_mc_per_sample_grad_fn,
)

__all__ = [
    "NeuralNetRegression",
    "create_regression_grad_fn",
    "create_regression_loglik_fn",
    "create_regression_per_sample_grad_fn",
    "NeuralNetModel",
    "create_nn_grad_fn",
    "create_nn_loglik_fn",
    "create_nn_per_sample_grad_fn",
    "generate_nn_stream_data",
    "MulticlassNeuralNetModel",
    "create_mc_grad_fn",
    "create_mc_loglik_fn",
    "create_mc_per_sample_grad_fn",
]
