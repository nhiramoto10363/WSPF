#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念ドリフト検知器(自前・軽量実装) — R2-4 ベースライン用

river 等の外部依存なしで、データストリーム学習の代表的な検知器を提供する:
  - ADWIN  : ADaptive WINdowing (Bifet & Gavaldà, 2007) の簡易版
  - PageHinkley : Page-Hinkley 検定(逐次変化点検知)

いずれも「予測誤差ストリーム(スカラー)」を受け取り、平均レベルの変化を
検知したら True を返す。ベースライン学習器はこの信号でモデルをリセットする。
"""

import math


class ADWIN:
    """
    ADWIN の簡易実装。

    直近の観測を可変長ウィンドウ W に保持し、W を全ての分割点で2分割して
    部分平均の差が Hoeffding 型の閾値 ε_cut を超えたら、古い側を破棄する
    (=変化検知)。厳密な指数ヒストグラム版ではなく素朴な実装だが、
    検知直後にウィンドウが縮むため実用上は軽量。
    """

    def __init__(self, delta=0.002, max_window=400):
        self.delta = delta
        self.max_window = max_window
        self.window = []

    def _try_shrink(self):
        changed = False
        shrunk = True
        while shrunk and len(self.window) >= 2:
            shrunk = False
            n = len(self.window)
            total = sum(self.window)
            s0 = 0.0
            for i in range(1, n):
                s0 += self.window[i - 1]
                n0, n1 = i, n - i
                mean0 = s0 / n0
                mean1 = (total - s0) / n1
                # 調和平均サイズと δ' = δ/n
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                delta_prime = self.delta / n
                eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / delta_prime))
                if abs(mean0 - mean1) > eps:
                    # 古い側 [0:i] を破棄
                    self.window = self.window[i:]
                    changed = True
                    shrunk = True
                    break
        return changed

    def update(self, x):
        """観測 x を追加し、変化を検知したら True。"""
        self.window.append(float(x))
        if len(self.window) > self.max_window:
            self.window.pop(0)
        return self._try_shrink()

    @property
    def width(self):
        return len(self.window)


class PageHinkley:
    """
    Page-Hinkley 検定(平均の上昇方向の変化を検知)。

    誤差平均の逐次推定 x̄_t を追跡し、累積偏差 m_T の最小値からの乖離
    PH_T = m_T − min m_t が λ を超えたら変化検知。

    Parameters
    ----------
    delta : float
        許容する平均の緩やかなドリフト(小さいほど敏感)
    lambda_ : float
        検知閾値
    alpha : float
        忘却係数(1 に近いほど長期記憶)
    """

    def __init__(self, delta=0.005, lambda_=5.0, alpha=0.9999):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.n = 0
        self.x_mean = 0.0
        self.m_t = 0.0
        self.min_m = 0.0

    def update(self, x):
        x = float(x)
        self.n += 1
        self.x_mean += (x - self.x_mean) / self.n
        self.m_t = self.alpha * self.m_t + (x - self.x_mean - self.delta)
        self.min_m = min(self.min_m, self.m_t)
        ph = self.m_t - self.min_m
        if ph > self.lambda_:
            self.reset()
            return True
        return False
