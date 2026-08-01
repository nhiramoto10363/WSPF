#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク (実験課題) パッケージ

各ベンチマークは共通の `Benchmark` プロトコル (base.py) を実装し、
モデル依存の関数群 (`build_functions`) とデータストリーム (`stream`) を供給する。
これにより evaluation/runner.py はベンチマークの中身を知らずに
PF / WSPF-A / WSPF-B / Oracle / SGD 系を同一ループで駆動できる。
"""

from src.benchmarks.base import Benchmark, StreamStep
from src.benchmarks.regression_switch import RegressionSwitchBenchmark
from src.benchmarks.gefcom import GefcomBenchmark
from src.benchmarks.email import EmailBenchmark
from src.benchmarks.insects import InsectsBenchmark

# 名前 → クラスの対応
_BENCHMARKS = {
    "regression": RegressionSwitchBenchmark,
    "gefcom": GefcomBenchmark,
    "email": EmailBenchmark,
    "insects": InsectsBenchmark,
}


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """名前からベンチマークを生成するファクトリ。

    Parameters
    ----------
    name : str
        "regression" | "gefcom" | "email"
    **kwargs
        各ベンチマークのコンストラクタ引数。

    Returns
    -------
    Benchmark
    """
    key = name.lower()
    if key not in _BENCHMARKS:
        raise ValueError(
            f"未知のベンチマーク '{name}'。利用可能: "
            f"{sorted(_BENCHMARKS.keys())}")
    return _BENCHMARKS[key](**kwargs)


__all__ = [
    "Benchmark",
    "StreamStep",
    "RegressionSwitchBenchmark",
    "GefcomBenchmark",
    "EmailBenchmark",
    "InsectsBenchmark",
    "get_benchmark",
]
