#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共有ランループ(runner) — 全ベンチマーク共通の唯一の駆動ループ

修正方針の中核: PF / WSPF-A / WSPF-B / Oracle / SGD 系(SGD / PH-SGD /
Window-SGD)を、ベンチマークの中身を知らずに **同一の prequential
(test-then-train)ループ** で駆動する。実験ファイルごとの独自ループを廃し、
評価・診断・出力を 1 か所に集約する。

prequential test-then-train (R1-12):
    各 StreamStep で
      1. まず現在の推定(θ もしくは重み付き粒子予測)で X_test を評価し、
      2. その後 X_train で更新する(baselines は .train、filters は .step)。
    straddles_switch が True のブロック(切替をまたぐ)は、集計側(region_mask)で
    全ての報告指標・switch整合解析から除外される。runner はマスクを記録するのみ。
    (Straddling blocks are excluded from all reported metrics and
     switch-aligned analyses.)

フィルタのシードは base_seed からのオフセット規約で決める:
    PF +1, WSPF-B +3, WSPF-A +5, Oracle +7, SGD 系 +10。
"""

from __future__ import annotations

import numpy as np

from src.filters import ParticleFilter, WSPF_B, WSPF_A, OraclePF
from src.baselines import OnlineSGD, PHSGD, WindowSGD

from . import metrics as M


# ======================================================================
# メソッド分類
# ======================================================================
SGD_METHODS = {"SGD", "PH-SGD", "Window-SGD"}
FILTER_METHODS = {"PF", "WSPF-A", "WSPF-B", "Oracle"}
ALL_METHODS = SGD_METHODS | FILTER_METHODS

# base_seed からのフィルタ/初期化シードオフセット規約
SEED_OFFSET = {
    "PF": 1,
    "WSPF-B": 3,
    "WSPF-A": 5,
    "Oracle": 7,
    "SGD": 10,
    "PH-SGD": 10,
    "Window-SGD": 10,
}


# ======================================================================
# 推定器の生成(method 文字列でディスパッチ)
# ======================================================================
def _build_estimator(method, benchmark, n_particles, params, seed, funcs,
                     filter_seed=None):
    """method に応じた推定器を生成して返す。

    filter_seed を指定すると、method 別オフセットの代わりにその値を
    フィルタ/初期化シードに使う(Oracle 比較の共通乱数 CRN 用)。
    """
    d = benchmark.param_dim
    eta = params["eta"]
    # sigma_sys(σ_cd)は粒子フィルタのみ必須。点推定ベースラインは持たない。
    sigma_sys = params.get("sigma_sys", 0.0)
    prior_std = params["prior_std"]
    prior_mean = params.get("prior_mean", 0.0)
    ess_ratio = params.get("ess_resample_ratio", 0.5)
    grad_clip = getattr(benchmark, "grad_clip_norm", None)
    fseed = (seed + SEED_OFFSET[method]) if filter_seed is None else filter_seed

    if method == "PF":
        return ParticleFilter(
            n_particles, d, eta=eta, sigma_sys=sigma_sys,
            prior_mean=prior_mean, prior_std=prior_std,
            ess_resample_ratio=ess_ratio, seed=fseed,
        )
    if method == "WSPF-B":
        return WSPF_B(
            n_particles, d, eta=eta, sigma_sys=sigma_sys,
            prior_mean=prior_mean, prior_std=prior_std,
            ess_resample_ratio=ess_ratio, grad_clip_norm=grad_clip, seed=fseed,
        )
    if method == "WSPF-A":
        return WSPF_A(
            n_particles, d, eta=eta, sigma_sys=sigma_sys,
            prior_mean=prior_mean, prior_std=prior_std,
            ess_resample_ratio=ess_ratio, grad_clip_norm=grad_clip,
            beta=params.get("beta", 0.9), seed=fseed,
        )
    if method == "Oracle":
        return OraclePF(
            n_particles, d, eta=eta, sigma_sys=sigma_sys,
            prior_mean=prior_mean, prior_std=prior_std,
            ess_resample_ratio=ess_ratio, grad_clip_norm=grad_clip, seed=fseed,
        )
    if method == "SGD":
        return OnlineSGD(d, eta, prior_std, funcs["grad_fn"],
                         seed=fseed, grad_clip_norm=grad_clip)
    if method == "PH-SGD":
        return PHSGD(d, eta, prior_std, funcs["grad_fn"],
                     seed=fseed, grad_clip_norm=grad_clip,
                     ph_delta=params.get("ph_delta", 0.005),
                     ph_lambda=params.get("ph_lambda", 5.0),
                     ph_alpha=params.get("ph_alpha", 0.9999))
    if method == "Window-SGD":
        return WindowSGD(d, eta, prior_std, funcs["grad_fn"],
                         window=params.get("window", 5),
                         n_passes=params.get("n_passes", 1),
                         seed=fseed, grad_clip_norm=grad_clip)
    raise ValueError(f"unknown method: {method!r} (valid: {sorted(ALL_METHODS)})")


# ======================================================================
# 予測の取得(baselines/filters を統一)
# ======================================================================
def _predict_particles_weights(method, estimator):
    """推定器から (particles(N,d), weights(N,)) を取り出す。"""
    if method in FILTER_METHODS:
        return estimator.particles, estimator.weights
    # 点推定: θ を 1 粒子として扱う
    theta = np.asarray(estimator.predict_theta()).reshape(1, -1)
    return theta, np.array([1.0])


# ======================================================================
# メインの実行関数
# ======================================================================
def run_method(method, benchmark, n_particles, params, seed,
               collect_diagnostics=True, filter_seed=None, max_steps=None):
    """
    1 メソッド × 1 シードを prequential ループで駆動する。

    Parameters
    ----------
    method : str
        {"SGD","PH-SGD","Window-SGD","PF","WSPF-A","WSPF-B","Oracle"}
    benchmark : Benchmark
    n_particles : int
        フィルタ系の粒子数(baselines では無視)。
    params : dict
        {"eta","sigma_sys","prior_std", (+"beta" for WSPF-A, ...)}
    seed : int
        base seed(データ生成・オフセット計算の基準)。
    collect_diagnostics : bool
        False の場合でも history はフィルタが常時収集するが、返却を省ける。

    Returns
    -------
    dict {
      "metrics": {metric_name: (T,) ndarray},
      "straddle_mask": (T,) bool,    # 切替をまたぐ(全報告指標・switch整合解析から除外)
      "switch_mask": (T,) bool,      # 既知の切替時点そのもの(回復解析の起点)
      "selection_mask": (T,) bool,   # 選択区間(HP選択の集計対象)
      "report_mask": (T,) bool,      # 報告区間(最終評価の集計対象)
      "history": dict or None,
      "n_resets": int,
      "regime_ids": (T,) int,
      "step_index": (T,) int,
      "predictions": {"y",("mean","std")|("probs")},  # report 区間サンプル単位
      "train_indices": list[ndarray], "test_indices": list[ndarray],
    }
    """
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method: {method!r}")

    funcs = benchmark.build_functions(seed)
    grad_fn = funcs["grad_fn"]
    per_sample_grad_fn = funcs.get("per_sample_grad_fn")
    loglik_fn = funcs["loglik_fn"]
    predict_fn = funcs["predict_fn"]
    task_type = benchmark.task_type
    is_reg = task_type == "regression"
    obs_sigma = float(funcs.get("obs_sigma", 0.0)) if is_reg else 0.0
    levels = (0.5, 0.8, 0.9, 0.95)

    estimator = _build_estimator(method, benchmark, n_particles, params,
                                 seed, funcs, filter_seed=filter_seed)

    # Oracle 用: per-step の真統計クロージャを供給する関数(回帰のみ)
    oracle_hook = getattr(benchmark, "oracle_stats_fn_for_step", None)
    _oseed = (seed + SEED_OFFSET.get(method, 0)) if filter_seed is None else filter_seed
    oracle_rng = np.random.default_rng(_oseed + 100)

    # 指標バッファ
    metric_lists = {}
    straddle = []
    switch = []
    selection = []
    report = []
    regime_ids = []
    step_indices = []
    # 予測・診断保存用 (P6, R2-5): report 区間のサンプル単位を蓄積
    rep_y, rep_mean, rep_std, rep_probs = [], [], [], []
    rep_block_step, rep_block_len = [], []   # 各報告ブロックの step_index と長さ
    train_idx_list, test_idx_list = [], []

    def _push(name, val):
        metric_lists.setdefault(name, []).append(float(val))

    for _i, stp in enumerate(benchmark.stream(seed)):
        # max_steps 指定時はループ自体をここで打ち切る(計時用に全ストリームを
        # 走らせない, 修正方針 compute)。
        if max_steps is not None and _i >= max_steps:
            break
        Xte, yte = stp.X_test, stp.y_test
        Xtr, ytr = stp.X_train, stp.y_train
        has_test = Xte is not None and np.asarray(Xte).shape[0] > 0

        # report 区間かつ straddle でないブロックのみ、サンプル単位予測を
        # 保存対象にする(較正 Brier/ECE/reliability から straddle を除外, F3)。
        is_reported_block = bool(stp.is_report_step) and not bool(stp.straddles_switch)

        # -------- 1) 評価(学習前) --------
        particles, weights = _predict_particles_weights(method, estimator)
        if has_test:
            pred_mean, pred_var = M.weighted_prediction(
                predict_fn, particles, weights, Xte)
            yte_arr = np.asarray(yte, dtype=np.float64).ravel()
            if is_reg:
                _push("mse", M.test_mse(yte_arr, pred_mean))
                _push("mae", M.test_mae(yte_arr, pred_mean))
                pred_std = M.prediction_std_with_noise(pred_var, obs_sigma)
                _push("nll", M.nll_gaussian(yte_arr, pred_mean, pred_std))
                _push("crps", M.crps_gaussian(yte_arr, pred_mean, pred_std))
                cw = M.coverage_and_width(yte_arr, pred_mean, pred_std, levels)
                for lvl in levels:
                    _push(f"coverage_{lvl:.2f}", cw[lvl]["coverage"])
                    _push(f"width_{lvl:.2f}", cw[lvl]["width"])
                primary_err = M.test_mse(yte_arr, pred_mean)
                if is_reported_block:
                    rep_y.append(yte_arr)
                    rep_mean.append(np.asarray(pred_mean, np.float64).ravel())
                    rep_std.append(np.asarray(pred_std, np.float64).ravel())
                    rep_block_step.append(int(stp.step_index))
                    rep_block_len.append(int(yte_arr.size))
            else:
                # 分類: predict_fn は確率(あるいは logit)。[0,1] 外は sigmoid。
                probs = pred_mean
                if np.any(probs < 0.0) or np.any(probs > 1.0):
                    probs = 1.0 / (1.0 + np.exp(-np.clip(probs, -60, 60)))
                hard = (probs > 0.5).astype(np.float64)
                _push("accuracy", M.accuracy(hard, yte_arr))
                _push("f1", M.f1(hard, yte_arr))
                _push("balanced_accuracy", M.balanced_accuracy(hard, yte_arr))
                _push("nll", M.nll_bernoulli(probs, yte_arr))
                _push("brier", float(np.mean((probs - yte_arr) ** 2)))
                primary_err = 1.0 - M.accuracy(hard, yte_arr)
                if is_reported_block:
                    rep_probs.append(np.asarray(probs, np.float64).ravel())
                    rep_y.append(yte_arr)
                    rep_block_step.append(int(stp.step_index))
                    rep_block_len.append(int(yte_arr.size))
        else:
            # test ブロックが空: NaN を記録して整列を保つ
            if is_reg:
                for name in ("mse", "mae", "nll", "crps"):
                    _push(name, float("nan"))
                for lvl in levels:
                    _push(f"coverage_{lvl:.2f}", float("nan"))
                    _push(f"width_{lvl:.2f}", float("nan"))
            else:
                for name in ("accuracy", "f1", "balanced_accuracy", "nll", "brier"):
                    _push(name, float("nan"))
            primary_err = None

        straddle.append(bool(stp.straddles_switch))
        switch.append(bool(getattr(stp, "is_switch_step", False)))
        selection.append(bool(getattr(stp, "is_selection_step", False)))
        report.append(bool(getattr(stp, "is_report_step", True)))
        regime_ids.append(-1 if stp.regime_id is None else int(stp.regime_id))
        step_indices.append(int(stp.step_index))
        train_idx_list.append(np.asarray(stp.train_indices, int))
        test_idx_list.append(np.asarray(stp.test_indices, int))

        # -------- 2) 更新(学習) --------
        if method in SGD_METHODS:
            estimator.train(Xtr, ytr)
            if primary_err is not None:
                estimator.observe_error(primary_err)
        elif method == "Oracle":
            oracle_stats_fn = None
            if oracle_hook is not None:
                oracle_stats_fn = oracle_hook(stp.step_index, oracle_rng)
            if oracle_stats_fn is None:
                # 回帰オラクルが未配線: フォールバックで ∇L=ĝ, Σ=0 相当
                def oracle_stats_fn(p, X, y):
                    g = per_sample_grad_fn(p, X, y).mean(axis=1)
                    d = p.shape[1]
                    return g, np.zeros((p.shape[0], d, d))
            estimator.step(Xtr, ytr, per_sample_grad_fn, loglik_fn,
                           oracle_stats_fn)
        elif method == "PF":
            estimator.step(Xtr, ytr, grad_fn, loglik_fn)
        else:  # WSPF-A / WSPF-B
            estimator.step(Xtr, ytr, per_sample_grad_fn, loglik_fn)

    metrics_out = {k: np.asarray(v, dtype=np.float64) for k, v in metric_lists.items()}
    history = None
    if collect_diagnostics and method in FILTER_METHODS:
        history = estimator.get_history()

    # report 区間のサンプル単位予測(較正・回復・保存用, R2-5/P6)
    def _cat(lst):
        return np.concatenate(lst) if lst else np.empty(0, np.float64)

    predictions = {"y": _cat(rep_y)}
    if is_reg:
        predictions["mean"] = _cat(rep_mean)
        predictions["std"] = _cat(rep_std)
    else:
        predictions["probs"] = _cat(rep_probs)
    # サンプル→ステップ対応(§6): 各報告ブロックの step_index/長さと、
    # flatten 済み予測に対する block 境界オフセット・サンプル別 step_index。
    block_step = np.asarray(rep_block_step, dtype=int)
    block_len = np.asarray(rep_block_len, dtype=int)
    predictions["block_step_index"] = block_step
    predictions["block_len"] = block_len
    predictions["offsets"] = np.concatenate([[0], np.cumsum(block_len)])
    predictions["pred_step_index"] = np.repeat(block_step, block_len) \
        if block_step.size else np.empty(0, int)

    return {
        "metrics": metrics_out,
        "straddle_mask": np.asarray(straddle, dtype=bool),
        "switch_mask": np.asarray(switch, dtype=bool),
        "selection_mask": np.asarray(selection, dtype=bool),
        "report_mask": np.asarray(report, dtype=bool),
        "history": history,
        "n_resets": int(getattr(estimator, "n_resets", 0)),
        "regime_ids": np.asarray(regime_ids, dtype=int),
        "step_index": np.asarray(step_indices, dtype=int),
        "predictions": predictions,
        "train_indices": train_idx_list,
        "test_indices": test_idx_list,
    }


def run_seeds(method, benchmark, n_particles, params, seeds,
              collect_diagnostics=True):
    """複数シードで run_method を実行し、per-seed 結果 dict のリストを返す。"""
    results = []
    for s in seeds:
        results.append(run_method(method, benchmark, n_particles, params, s,
                                  collect_diagnostics=collect_diagnostics))
    return results
