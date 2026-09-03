"""Benchmarks (learning tasks).

Each benchmark implements the `Benchmark` protocol in base.py, supplying
model-dependent functions via build_functions() and a data stream via
stream(). This lets evaluation/runner.py drive PF, WSPF-A, WSPF-B, the oracle
filter and the point-estimate baselines through one loop, without knowing
anything about the task.
"""

from src.benchmarks.base import Benchmark, StreamStep
from src.benchmarks.regression_switch import RegressionSwitchBenchmark
from src.benchmarks.email import EmailBenchmark
from src.benchmarks.insects import InsectsBenchmark

_BENCHMARKS = {
    "regression": RegressionSwitchBenchmark,
    "email": EmailBenchmark,
    "insects": InsectsBenchmark,
}


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """Construct a benchmark by name ("regression" / "email" / "insects")."""
    key = name.lower()
    if key not in _BENCHMARKS:
        raise ValueError(
            f"unknown benchmark '{name}'; available: "
            f"{sorted(_BENCHMARKS.keys())}")
    return _BENCHMARKS[key](**kwargs)


__all__ = [
    "Benchmark",
    "StreamStep",
    "RegressionSwitchBenchmark",
    "EmailBenchmark",
    "InsectsBenchmark",
    "get_benchmark",
]
