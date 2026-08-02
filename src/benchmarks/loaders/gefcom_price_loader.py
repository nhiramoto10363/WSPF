#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEFCom2014 Price track (GEFCom2014-P) のローダー

== データ定義 (実験の正式定義; 論文にもこの通り記載する) ==
- トラック: Price (Hong et al., 2016, IJF)。単一ゾーン (ZONEID=1), 米国系統,
  時別、電力の zonal price を予測する。
- 使用ファイル: Task 15 の統合ファイル (全履歴 2011-01-01 00:00 〜
  2013-12-17 23:00 を含む)。末尾 24h の Zonal Price 空欄は Solution to
  Task 15 で補完済み (scripts でなく既作成の price15.csv)。
    price15.csv : ZONEID, timestamp, Forecasted Total Load,
                  Forecasted Zonal Load, Zonal Price
- タスク: 点予測回帰 (主指標は確率的評価 = pinball/CRPS/coverage, 副が MSE/MAE)。
  y = 変換後 Zonal Price (§D1), X = 負荷 2 変数 + 時刻/曜日 + load ratio +
  週末フラグ (input_dim=8)。prequential test-then-train (Solar と同一)。
  Solar と役割分担: Solar=漸進的季節ドリフト・大 d、Price=急峻スパイク・
  確率評価が主・中庸 d (d=321)。
- 目的変数の変換 (D1): y' = asinh(price / s)。s は **選択区間** の
  ロバストスケール 1.4826 * MAD(price)。負値/ゼロが無くても大値を log 的に
  圧縮しガウス観測尤度をスパイクから守る。全指標は変換後スケールで一貫報告
  (逆変換報告は runner 改修が要るため初版はしない, 論文に明記)。
- 特徴量 (input_dim=8, 価格ラグは入れない = D2):
    Forecasted Zonal Load, Forecasted Total Load, load ratio(Zonal/Total),
    hour sin/cos, day-of-week sin/cos, is_weekend
  ※Solar 固有の「蓄積変数の非蓄積化」「昼間フィルタ」は Price では不要 (持ち込まない)。
- タイムスタンプ形式: `MMDDYYYY H:MM` (Solar の YYYYMMDD とは別。専用パーサ)。
  DST 由来の重複 (2013-03-10 01:00 が 2 行) と 02:00 欠落が各 1 件あるが、
  安定ソートで許容する (Price は block 処理を持たない)。
- リーク除去 (R1-13): 特徴量の標準化統計・目的変数の変換スケール s・σ_obs は
  すべて選択区間 (ts < select_end_ts) のみで fit。将来 (報告区間) は一切見ない。
- ドリフトの性格: 既知スイッチ点なし。switch-aligned 解析は行わず、スパイク・
  エピソード (選択区間の変換後残差 q99 超過ブロック, D4) 別誤差を集計側で報告する。
"""

import numpy as np
from datetime import datetime, timedelta

# price15.csv の実列名
COL_ZONE = "ZONEID"
COL_TS = "timestamp"
COL_TOTAL_LOAD = "Forecasted Total Load"
COL_ZONAL_LOAD = "Forecasted Zonal Load"
COL_PRICE = "Zonal Price"

# 特徴量の並び (input_dim = 8)。標準化は選択区間統計で行う。
FEATURE_NAMES = [
    "zonal_load", "total_load", "load_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
]


def _parse_ts_price(s):
    """GEFCom2014-Price の 'MMDDYYYY H:MM' / 'MMDDYYYY HH:MM' を datetime に。

    24:00 表記が来た場合は翌日 00:00 に丸める (Solar パーサと同じ扱い)。
    """
    s = s.strip().strip('"')
    date_part, time_part = s.split()
    h, m = time_part.split(":")
    base = datetime.strptime(date_part, "%m%d%Y")
    hh = int(h)
    if hh == 24:
        return base + timedelta(days=1, minutes=int(m))
    return base + timedelta(hours=hh, minutes=int(m))


def _read_csv_dict(path):
    """ヘッダ付き CSV を {colname: list[str]} で読む (依存なし, CRLF 許容)。"""
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        header = [c.strip().strip('"') for c in
                  fp.readline().strip().split(",")]
        cols = {c: [] for c in header}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            for c, v in zip(header, parts):
                cols[c].append(v.strip())
    return cols


class GefcomPriceLoader:
    """
    GEFCom2014 Price の 1 ストリームを保持する (単一ゾーン)。

    Attributes (Solar ローダーと同一契約)
    ----------
    X : ndarray (n_samples, 8)   標準化済み特徴 (負荷2 + ratio + 時刻/曜日 + 週末)
    y : ndarray (n_samples,)     変換後価格 asinh(price / s)
    timestamps : list of datetime
    select_mask : ndarray bool   選択区間 (ts < select_end_ts)
    price_raw : ndarray          変換前の生 price (診断用)
    target_scale_ : float        D1 のロバストスケール s (選択区間で fit)
    scale_mean_, scale_std_ : ndarray  特徴標準化統計 (選択区間で fit)
    noise_std_ : float or None   set_noise_std() で外部から設定
    """

    def __init__(self, data_path, select_end_ts="2012-01-01",
                 target_transform="asinh", seed=42):
        self.select_end_ts = datetime.fromisoformat(select_end_ts)
        self.target_transform = str(target_transform)
        self.seed = int(seed)

        raw = _read_csv_dict(data_path)
        for c in (COL_TS, COL_TOTAL_LOAD, COL_ZONAL_LOAD, COL_PRICE):
            if c not in raw:
                raise KeyError(
                    f"列 '{c}' が {data_path} にありません。price15.csv の "
                    "列名を確認してください。")

        # --- パース & 時刻ソート (DST 重複は安定ソートで許容) ---
        ts = [_parse_ts_price(t) for t in raw[COL_TS]]
        order = np.argsort(np.asarray(ts), kind="stable")
        ts = [ts[i] for i in order]

        def _col(name):
            arr = raw[name]
            return np.asarray([float(arr[i]) for i in order], dtype=np.float64)

        total_load = _col(COL_TOTAL_LOAD)
        zonal_load = _col(COL_ZONAL_LOAD)
        price = _col(COL_PRICE)

        warm = np.asarray([t < self.select_end_ts for t in ts])
        if warm.sum() < 24 * 30:
            raise ValueError(
                "選択区間が短すぎます (select_end_ts を確認してください)。")

        # --- 特徴量 (input_dim = 8) ---
        hours = np.asarray([t.hour for t in ts], dtype=np.float64)
        dows = np.asarray([t.weekday() for t in ts], dtype=np.float64)  # 月=0
        load_ratio = zonal_load / np.where(total_load == 0.0, np.nan,
                                           total_load)
        # total_load はすべて正 (Phase 0 実測: <=0 は 0 件) だが安全側で処理
        load_ratio = np.where(np.isfinite(load_ratio), load_ratio, 0.0)
        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)
        dow_sin = np.sin(2 * np.pi * dows / 7.0)
        dow_cos = np.cos(2 * np.pi * dows / 7.0)
        is_weekend = (dows >= 5).astype(np.float64)

        feats = np.column_stack([
            zonal_load, total_load, load_ratio,
            hour_sin, hour_cos, dow_sin, dow_cos, is_weekend,
        ])

        # --- 標準化: 選択区間統計のみで fit (リークフリー) ---
        mu = feats[warm].mean(axis=0)
        sd = feats[warm].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        # 注意: per-feature 標準化のみ。whitening / PCA は行わない
        # (WSPF-A が利用する異方的勾配共分散構造を保存するため, Solar と同方針)。
        self.scale_mean_, self.scale_std_ = mu, sd
        self.X = (feats - mu) / sd

        # --- 目的変数の変換 (D1): 選択区間の MAD ベースのスケールで asinh ---
        s = self._fit_target_scale(price[warm])
        self.target_scale_ = s
        self.y = self._transform_target(price, s)

        self.price_raw = price
        self.timestamps = ts
        self.select_mask = warm
        self.n_samples, self.n_features = self.X.shape
        self.noise_std_ = None

    # ------------------------------------------------------------------
    def _fit_target_scale(self, price_sel):
        """選択区間の price から D1 のロバストスケール s = 1.4826*MAD を得る。"""
        med = np.median(price_sel)
        mad = np.median(np.abs(price_sel - med))
        s = 1.4826 * mad
        return float(max(s, 1e-6))

    def _transform_target(self, price, s):
        if self.target_transform == "asinh":
            return np.arcsinh(price / s)
        if self.target_transform in ("none", "identity", None):
            return price.astype(np.float64)
        if self.target_transform == "log":
            # 参考: 負値なしを前提。price>0 のみで使用可。
            return np.log(np.maximum(price, 1e-6) / s)
        raise ValueError(
            f"未知の target_transform '{self.target_transform}'。")

    def inverse_transform_target(self, y):
        """変換後 y を生 price スケールへ戻す (診断・逆変換報告の任意利用)。"""
        s = self.target_scale_
        y = np.asarray(y, dtype=np.float64)
        if self.target_transform == "asinh":
            return np.sinh(y) * s
        if self.target_transform in ("none", "identity", None):
            return y
        if self.target_transform == "log":
            return np.exp(y) * s
        raise ValueError(
            f"未知の target_transform '{self.target_transform}'。")

    def set_noise_std(self, s):
        self.noise_std_ = float(s)

    def print_summary(self, emit=print):
        emit(f"  GEFCom2014-P (single zone): n={self.n_samples}, "
             f"features={self.n_features}")
        emit(f"  period: {self.timestamps[0]} .. {self.timestamps[-1]}")
        emit(f"  select region: ts < {self.select_end_ts} "
             f"({int(self.select_mask.sum())} samples)")
        emit(f"  target transform: {self.target_transform}, "
             f"scale s={self.target_scale_:.4f}")
        pr = self.price_raw
        emit(f"  raw price: median={np.median(pr):.2f}, "
             f"q99={np.percentile(pr, 99):.2f}, max={pr.max():.2f}")
