#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データローダー群 (benchmarks サブパッケージ)

旧 `src/data/` から移設。各ローダーは numpy のみに依存する自己完結実装で、
ARFF / CSV を手動パースし、リークフリーな前処理 (PCA / 標準化 / 昼間フィルタ) を
提供する。
"""

from .email_loader import EmailDataLoader
from .gefcom_solar_loader import GefcomSolarLoader
from .gefcom_price_loader import GefcomPriceLoader
from .insects_loader import InsectsDataLoader

__all__ = [
    "EmailDataLoader",
    "GefcomSolarLoader",
    "GefcomPriceLoader",
    "InsectsDataLoader",
]
