#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEFCom2014 Solar track (GEFCom2014-S) のローダー

== データ定義 (実験の正式定義; 論文にもこの通り記載する) ==
- トラック: Solar (Hong et al., 2016, IJF)。3 プラント (ZONEID 1-3),
  豪州, 時別, 電力は各プラント容量で [0,1] に正規化済み。
- 使用ファイル: Task 15 の統合ファイル (全履歴 2012-04-01 01:00 〜
  2014-07-01 00:00 を含む)。
    predictors15.csv : ZONEID, TIMESTAMP, VAR78..VAR228 (ECMWF NWP 12 変数)
    train15.csv      : ZONEID, TIMESTAMP, POWER
  ★手元のファイル名・列名と要照合。単一の結合済み CSV でも可
  (POWER 列があれば merged とみなす)。
- タスク: 点予測回帰。y = POWER, X = NWP 12 変数 + 時刻 (sin/cos)。
  prequential test-then-train, 主指標 MSE (副 MAE)。
  確率的評価 (pinball) からの転用は論文で正当化を一段落記載する。
- 蓄積変数の非蓄積化: VAR169 (SSRD), VAR175 (STRD), VAR178 (TSR),
  VAR228 (TP) は ECMWF の予報開始時刻からの累積値。日次予報ブロック
  (リード 1-24h = 01:00-翌00:00) 内で階差を取り時別値に変換する。
  ★手元データが既に時別値なら DEACCUMULATE_VARS = [] にすること★
- 夜間の扱い: warm-up 区間の時刻別平均 POWER が閾値以上の時刻
  (hour-of-day) のみを「昼間」として残す。集合は warm-up で固定
  (リークフリー)。夜間はターゲットがゼロ強制であり、点予測回帰の
  評価を歪めるため除外する。
- ドリフトの性格: 既知スイッチ点のない漸進的な季節ドリフト
  (regression sim / INSECTS の abrupt と相補的)。switch-aligned 解析は
  行わず、月別誤差の時系列で報告する。
- リーク除去: 標準化・昼間時刻集合・σ_obs・ハイパラ選択はすべて
  選択区間 (SELECT_END_TS より前) のみで決定。
"""

import numpy as np
from datetime import datetime, timedelta

NWP_VARS = ["VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165",
            "VAR166", "VAR167", "VAR169", "VAR175", "VAR178", "VAR228"]
DEACCUMULATE_VARS = ["VAR169", "VAR175", "VAR178", "VAR228"]

DAY_POWER_THRESH = 0.01   # warm-up 時刻別平均 POWER の昼間判定閾値


def _parse_ts(s):
    """GEFCom2014 の 'YYYYMMDD H:MM' / 'YYYYMMDD HH:MM' を datetime に"""
    s = s.strip().strip('"')
    date_part, time_part = s.split()
    h, m = time_part.split(":")
    base = datetime.strptime(date_part, "%Y%m%d")
    # 24:00 表記対策
    hh = int(h)
    if hh == 24:
        return base + timedelta(days=1)
    return base + timedelta(hours=hh, minutes=int(m))


def _read_csv_dict(path):
    """ヘッダ付き CSV を {colname: list[str]} で読む (依存なし)"""
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        header = [c.strip().strip('"') for c in
                  fp.readline().strip().split(",")]
        cols = {c: [] for c in header}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            for c, v in zip(header, line.split(",")):
                cols[c].append(v.strip())
    return cols


class GefcomSolarLoader:
    """
    1 ゾーン分のストリームを保持する。

    Attributes
    ----------
    X : ndarray (n_samples, 14)  標準化済み特徴 (NWP12 + hour sin/cos)
    y : ndarray (n_samples,)     POWER (正規化済み, 標準化しない)
    timestamps : list of datetime
    select_mask : ndarray bool   選択区間 (ts < select_end_ts)
    day_hours : set of int       昼間として残した hour-of-day
    noise_std_ : float or None   set_noise_std() で外部から設定
    """

    def __init__(self, predictors_path, zone=1, train_path=None,
                 select_end_ts="2012-10-01"):
        self.zone = int(zone)
        self.select_end_ts = datetime.fromisoformat(select_end_ts)

        pred = _read_csv_dict(predictors_path)
        if "POWER" in pred and train_path is None:
            merged = pred
        else:
            if train_path is None:
                raise ValueError(
                    "predictors に POWER 列がありません。train_path "
                    "(train15.csv) を指定してください。")
            tr = _read_csv_dict(train_path)
            key_tr = {}
            for i in range(len(tr["TIMESTAMP"])):
                key_tr[(tr["ZONEID"][i], tr["TIMESTAMP"][i])] = \
                    tr["POWER"][i]
            # predictors 側に既に POWER 列がある場合は train15 の値で
            # 上書きする。pred のキー列挙で POWER を二重追記しないよう、
            # 特徴列 (POWER 以外) のみをコピーし POWER は train から付与する。
            pred_cols = [c for c in pred.keys() if c != "POWER"]
            merged = {c: [] for c in pred_cols + ["POWER"]}
            for i in range(len(pred["TIMESTAMP"])):
                k = (pred["ZONEID"][i], pred["TIMESTAMP"][i])
                if k not in key_tr:      # POWER の無い将来予報行は捨てる
                    continue
                for c in pred_cols:
                    merged[c].append(pred[c][i])
                merged["POWER"].append(key_tr[k])

        # ゾーン抽出
        zid = np.asarray([int(float(z)) for z in merged["ZONEID"]])
        sel = np.nonzero(zid == self.zone)[0]
        if sel.size == 0:
            raise ValueError(f"ZONEID={self.zone} の行がありません。")

        ts = [_parse_ts(merged["TIMESTAMP"][i]) for i in sel]
        order = np.argsort(np.asarray(ts))
        sel = sel[order]
        ts = [ts[i] for i in order]

        for v in NWP_VARS:
            if v not in merged:
                raise KeyError(f"列 '{v}' がありません。NWP_VARS を "
                               "手元データの列名に合わせてください。")
        raw = np.asarray(
            [[float(merged[v][i]) for v in NWP_VARS] for i in sel],
            dtype=np.float64)
        power = np.asarray([float(merged["POWER"][i]) for i in sel])

        # --- 蓄積変数の非蓄積化 (日次予報ブロック内で階差) ---
        # ブロック = (ts − 1h) の暦日 (リード 1-24h が同一ブロック)
        block = np.asarray([(t - timedelta(hours=1)).toordinal()
                            for t in ts])
        for v in DEACCUMULATE_VARS:
            j = NWP_VARS.index(v)
            col = raw[:, j].copy()
            out = col.copy()
            out[1:] = np.where(block[1:] == block[:-1],
                               col[1:] - col[:-1], col[1:])
            # 数値誤差による僅かな負値はゼロに
            raw[:, j] = np.maximum(out, 0.0)

        # --- 昼間フィルタ (warm-up の時刻別平均 POWER で決定・固定) ---
        hours = np.asarray([t.hour for t in ts])
        warm = np.asarray([t < self.select_end_ts for t in ts])
        if warm.sum() < 24 * 30:
            raise ValueError("選択区間が短すぎます (select_end_ts を確認)。")
        day_hours = set()
        for h in range(24):
            m = warm & (hours == h)
            if m.any() and power[m].mean() >= DAY_POWER_THRESH:
                day_hours.add(h)
        keep = np.asarray([h in day_hours for h in hours])
        self.day_hours = day_hours

        ts = [t for t, k in zip(ts, keep) if k]
        raw = raw[keep]
        power = power[keep]
        hours = hours[keep]
        warm = warm[keep]

        # --- 特徴量: NWP12 + hour sin/cos。標準化は選択区間統計のみ ---
        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)
        feats = np.column_stack([raw, hour_sin, hour_cos])
        mu = feats[warm].mean(axis=0)
        sd = feats[warm].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        # 注意: per-feature 標準化のみ。whitening / PCA は行わない
        # (WSPF-A が利用する異方的勾配共分散構造を保存するため)。
        self.scale_mean_, self.scale_std_ = mu, sd

        self.X = (feats - mu) / sd
        self.y = power
        self.timestamps = ts
        self.select_mask = warm
        self.n_samples, self.n_features = self.X.shape
        self.noise_std_ = None

    def set_noise_std(self, s):
        self.noise_std_ = float(s)

    def print_summary(self, emit=print):
        emit(f"  GEFCom2014-S zone {self.zone}: n={self.n_samples} "
             f"(day-filtered), features={self.n_features}")
        emit(f"  period: {self.timestamps[0]} .. {self.timestamps[-1]}")
        emit(f"  select region: ts < {self.select_end_ts} "
             f"({int(self.select_mask.sum())} samples)")
        emit(f"  day hours ({len(self.day_hours)}): "
             f"{sorted(self.day_hours)}")
        emit(f"  POWER: mean={self.y.mean():.4f}, "
             f"warm-up mean={self.y[self.select_mask].mean():.4f}")
