#!/usr/bin/env python3
"""Build the supplementary LaTeX tables from finalized outputs.

Reads only the artifacts under outputs/{regression,email,insects}/ and never
re-runs an experiment. Writes booktabs fragments to
outputs/supplementary_tables/ - the pieces the supplementary material
\\inputs - together with the reliability figures.

The aggregation follows the same convention as the main text (runner.py):
metrics are computed per test block, averaged over blocks within a seed, then
averaged across seeds. Only reporting, non-straddling blocks are stored in
predictions.npz, so reading from there applies the exclusions automatically.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
SUPP = os.path.join(OUTPUTS, "supplementary_tables")

BENCH = ["regression", "email", "insects"]
BENCH_LABEL = {"regression": "Regression", "email": "Email", "insects": "INSECTS"}
# Row order, matching the main text
METHOD_ORDER = ["PF", "WSPF-A", "WSPF-B", "SGD", "PH-SGD", "Window-SGD", "Oracle"]
DIR2METHOD = {
    "pf": "PF", "wspf_a": "WSPF-A", "wspf_b": "WSPF-B",
    "sgd": "SGD", "ph_sgd": "PH-SGD", "window_sgd": "Window-SGD",
}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
# Map the Greek letters and symbols that appear in the CSVs to LaTeX math
UNICODE_MAP = {
    "\u03b8": r"$\theta$", "\u03c1": r"$\rho$", "\u03c3": r"$\sigma$",
    "\u03b7": r"$\eta$", "\u03b2": r"$\beta$", "\u03bb": r"$\lambda$",
    "\u03b1": r"$\alpha$", "\u03b4": r"$\delta$", "\u03a3": r"$\Sigma$",
    "\u2192": r"$\rightarrow$", "\u2264": r"$\le$", "\u2265": r"$\ge$",
}


def esc(s: str) -> str:
    out = str(s)
    for k, v in UNICODE_MAP.items():
        out = out.replace(k, v)
    out = out.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    # Fail loudly on any unmapped non-ASCII character
    leftover = [c for c in out if ord(c) > 127]
    if leftover:
        raise ValueError(f"unmapped non-ASCII {leftover!r} in {s!r}")
    return out


def num(x, fmt="{:.4f}"):
    if x is None or x == "" or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        m, e = f"{v:.2e}".split("e")
        return f"${m}\\times 10^{{{int(e)}}}$"
    return fmt.format(v)


def order_key(m):
    return METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)


LONG_THRESHOLD = 24


def write_table(name, colspec, header, rows, midrules=(),
                caption="", label="", size=r"\footnotesize", colsep="3pt"):
    """Write a table float, or a longtable when there are many rows."""
    head = " & ".join(header) + r" \\"
    body = []
    for i, r in enumerate(rows):
        if i in midrules:
            body.append(r"\midrule")
        body.append(" & ".join(r) + r" \\")
    lbl = label or os.path.splitext(name)[0]

    if len(rows) <= LONG_THRESHOLD:
        lines = [r"\begin{table}[htbp]", r"\centering", size,
                 f"\\setlength{{\\tabcolsep}}{{{colsep}}}",
                 f"\\caption{{{caption}}}", f"\\label{{tab:{lbl}}}",
                 f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", r"\toprule",
                 head, r"\midrule", *body,
                 r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    else:
        lines = ["{" + size, f"\\setlength{{\\tabcolsep}}{{{colsep}}}",
                 f"\\begin{{longtable}}{{@{{}}{colspec}@{{}}}}",
                 f"\\caption{{{caption}}}\\label{{tab:{lbl}}}\\\\",
                 r"\toprule", head, r"\midrule", r"\endfirsthead",
                 f"\\multicolumn{{{len(header)}}}{{@{{}}l}}"
                 r"{\emph{(continued)}} \\",
                 r"\toprule", head, r"\midrule", r"\endhead",
                 r"\bottomrule", r"\endlastfoot", *body,
                 r"\end{longtable}", "}"]
    path = os.path.join(SUPP, name)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    kind = "table" if len(rows) <= LONG_THRESHOLD else "longtable"
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(rows)} rows, {kind})")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------
# S1: selected hyper-parameters
# ----------------------------------------------------------------------
HP_COLS = [("eta", r"$\eta$"), ("sigma_sys", r"$\sigma_{\mathrm{cd}}$"),
           ("prior_std", r"$\sigma_0$"), ("beta", r"$\beta$"),
           ("window", "$W$"), ("n_passes", "$K$"),
           ("ph_delta", r"$\delta$"), ("ph_lambda", r"$\lambda$"),
           ("ph_alpha", r"$\alpha$")]


def build_s1():
    rows, mids = [], set()
    for b in BENCH:
        rec = read_csv(os.path.join(OUTPUTS, b, "selected_hp_table.csv"))
        if not rec:
            continue
        if rows:
            mids.add(len(rows))
        rec.sort(key=lambda r: order_key(r["method"]))
        for j, r in enumerate(rec):
            rows.append([BENCH_LABEL[b] if j == 0 else "",
                         esc(r["method"])] +
                        [(r.get(c) or "---") for c, _ in HP_COLS])
    write_table("s1_selected_hp.tex", "ll" + "r" * len(HP_COLS),
                ["Benchmark", "Method"] + [t for _, t in HP_COLS], rows, mids,
                size=r"\scriptsize",
                caption="Hyper-parameters selected by the common protocol for every method and benchmark. Blank entries do not apply to the method. $W$ and $K$ are the window width and the number of passes of Window-SGD; $\\delta$, $\\lambda$ and $\\alpha$ are the drift tolerance, detection threshold and forgetting factor of PH-SGD.")


# ----------------------------------------------------------------------
# S2: the complete pairwise test table
# ----------------------------------------------------------------------
METRIC_LABEL = {"mse_paired": "MSE", "f1_paired": "$F_1$",
                "nll_paired": "NLL", "nll_mixture_paired": "NLL (mixture)"}


def build_s2():
    rows, mids = [], set()
    for b in BENCH:
        rec = [r for r in read_csv(os.path.join(OUTPUTS, b, "table_paired_tests.csv"))
               if r["metric"] in METRIC_LABEL]
        if not rec:
            continue
        for exp in ["main", "oracle"]:
            sub = [r for r in rec if r["experiment"] == exp]
            if not sub:
                continue
            if rows:
                mids.add(len(rows))
            tag = BENCH_LABEL[b] + ("" if exp == "main" else " (oracle)")
            sub.sort(key=lambda r: (r["metric"], r["method"]))
            for j, r in enumerate(sub):
                rows.append([tag if j == 0 else "",
                             esc(r["method"].replace("_vs_", " vs.\\ ")),
                             METRIC_LABEL[r["metric"]],
                             num(r["mean_difference"]), num(r["std_difference"]),
                             num(r["paired_t_p"]), num(r["paired_t_p_holm"]),
                             num(r["wilcoxon_p"]), num(r["wilcoxon_p_holm"])])
    write_table("s2_paired_tests.tex", "lllrrrrrr",
                ["Benchmark", "Comparison", "Metric", r"$\Delta$",
                 r"sd($\Delta$)", "$t$ $p$", "$t$ $p_{\\mathrm{Holm}}$",
                 "$W$ $p$", "$W$ $p_{\\mathrm{Holm}}$"], rows, mids,
                size=r"\scriptsize",
                caption="Complete set of paired comparisons. $\\Delta$ is the mean difference over the ten evaluation seeds and sd($\\Delta$) its standard deviation. Both a paired $t$-test and a Wilcoxon signed-rank test are reported, raw and after Holm--Bonferroni correction across all method pairs for that benchmark, experiment and metric. The values quoted in the main text are the Holm-corrected $t$-test column.")


# ----------------------------------------------------------------------
# Shared: rebuild the classification metrics from predictions.npz
# ----------------------------------------------------------------------
def _per_block_classification(bench):
    """Classification metrics under the main-text convention.

    Returns {method: {metric: (mean, sd)}}.
    """
    runs = os.path.join(OUTPUTS, bench, "main", "runs")
    per = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(os.path.join(runs, "*_seed*"))):
        base = os.path.basename(d)
        meth = DIR2METHOD.get(base.rsplit("_seed", 1)[0])
        f = os.path.join(d, "predictions.npz")
        if meth is None or not os.path.exists(f):
            continue
        z = np.load(f)
        y = z["y"].astype(int)
        probs = z["probs"]
        off = z["offsets"]
        multiclass = probs.ndim == 2
        C = probs.shape[1] if multiclass else 2
        hard = probs.argmax(1).astype(int) if multiclass else (probs > 0.5).astype(int)
        acc, prec, rec, f1, bal = [], [], [], [], []
        for i in range(len(off) - 1):
            yy, pp = y[off[i]:off[i + 1]], hard[off[i]:off[i + 1]]
            if yy.size == 0:
                continue
            acc.append(float(np.mean(pp == yy)))
            if multiclass:                       # macro average (INSECTS)
                ps, rs, fs = [], [], []
                for c in range(C):
                    tp = np.sum((pp == c) & (yy == c))
                    fp = np.sum((pp == c) & (yy != c))
                    fn = np.sum((pp != c) & (yy == c))
                    p_ = tp / (tp + fp) if tp + fp else 0.0
                    r_ = tp / (tp + fn) if tp + fn else 0.0
                    ps.append(p_); rs.append(r_)
                    fs.append(0.0 if p_ + r_ == 0 else 2 * p_ * r_ / (p_ + r_))
                prec.append(np.mean(ps)); rec.append(np.mean(rs))
                f1.append(np.mean(fs)); bal.append(np.mean(rs))
            else:                                # positive class (email)
                tp = np.sum((pp == 1) & (yy == 1)); fp = np.sum((pp == 1) & (yy == 0))
                fn = np.sum((pp == 0) & (yy == 1)); tn = np.sum((pp == 0) & (yy == 0))
                p_ = tp / (tp + fp) if tp + fp else 0.0
                r_ = tp / (tp + fn) if tp + fn else 0.0
                tnr = tn / (tn + fp) if tn + fp else 0.0
                prec.append(p_); rec.append(r_); bal.append(0.5 * (r_ + tnr))
                f1.append(0.0 if p_ + r_ == 0 else 2 * p_ * r_ / (p_ + r_))
        per[meth]["accuracy"].append(np.mean(acc))
        per[meth]["precision"].append(np.mean(prec))
        per[meth]["recall"].append(np.mean(rec))
        per[meth]["f1"].append(np.mean(f1))
        per[meth]["balanced_accuracy"].append(np.mean(bal))
    return {m: {k: (float(np.mean(v)), float(np.std(v, ddof=1)))
                for k, v in d.items()} for m, d in per.items()}


# ----------------------------------------------------------------------
# S3: per-seed means and standard deviations
# ----------------------------------------------------------------------
def build_s3():
    """S3a: regression levels (MSE, NLL, CRPS, coverage, width)."""
    rows = []
    ctx = {r["method"]: r for r in
           read_csv(os.path.join(OUTPUTS, "regression", "table_context.csv"))
           if r["source"] == "main"}
    cal = read_csv(os.path.join(OUTPUTS, "regression", "calibration",
                                "calibration_report.csv"))

    def cal_get(meth, metric, level=""):
        for r in cal:
            if r["method"] == meth and r["metric"] == metric and r["level"] == level:
                return r["mean"], r["std"]
        return None, None

    for meth in sorted(ctx, key=order_key):
        r = ctx[meth]
        cells = [f"{num(r['mse_mean'])} ({num(r['mse_std'])})",
                 f"{num(r['nll_mean'])} ({num(r['nll_std'])})"]
        for metric, lvl in [("crps", ""), ("coverage", "0.90"), ("width", "0.90")]:
            m_, s_ = cal_get(meth, metric, lvl)
            cells.append("---" if m_ is None else f"{num(m_)} ({num(s_)})")
        rows.append([esc(meth)] + cells)
    write_table("s3_per_seed_sd.tex", "l" + "r" * 5,
                ["Method", "MSE", "NLL", "CRPS", "Coverage@0.90", "Width@0.90"],
                rows,
                caption="Regression benchmark: per-seed means with standard "
                        "deviations in parentheses, over the ten evaluation "
                        "seeds. The three point estimators share an identical "
                        "predictive width at every nominal level, because their "
                        "intervals carry observation noise alone.")


def build_s3b():
    """Calibration for classification, kept to four columns."""
    rows, mids = [], set()
    for b in ["email", "insects"]:
        cal = read_csv(os.path.join(OUTPUTS, b, "calibration", "calibration_report.csv"))
        if not cal:
            continue
        if rows:
            mids.add(len(rows))
        meths = sorted({r["method"] for r in cal}, key=order_key)
        for j, meth in enumerate(meths):
            def g(metric):
                for r in cal:
                    if r["method"] == meth and r["metric"] == metric:
                        return f"{num(r['mean'])} ({num(r['std'])})"
                return "---"
            rows.append([BENCH_LABEL[b] if j == 0 else "", esc(meth), g("ece"), g("brier")])
    write_table("s3b_calibration.tex", "llrr",
                ["Benchmark", "Method", "ECE", "Brier"], rows, mids,
                caption="Calibration on the two classification streams: expected calibration error and Brier score, each computed per seed and then averaged, never pooled. Standard deviations across seeds in parentheses.")


def build_s3c():
    """S3c: classification levels (F1, accuracy, NLL)."""
    rows, mids = [], set()
    for b in ["email", "insects"]:
        ctx = {r["method"]: r for r in
               read_csv(os.path.join(OUTPUTS, b, "table_context.csv"))
               if r["source"] == "main"}
        cls = _per_block_classification(b)
        if not ctx:
            continue
        if rows:
            mids.add(len(rows))
        for j, meth in enumerate(sorted(ctx, key=order_key)):
            r = ctx[meth]
            a = cls.get(meth, {}).get("accuracy")
            rows.append([BENCH_LABEL[b] if j == 0 else "", esc(meth),
                         f"{num(r['f1_mean'])} ({num(r['f1_std'])})",
                         "---" if a is None else f"{num(a[0])} ({num(a[1])})",
                         f"{num(r['nll_mean'])} ({num(r['nll_std'])})"])
    write_table("s3c_classification_levels.tex", "ll" + "r" * 3,
                ["Benchmark", "Method", "$F_1$", "Accuracy", "NLL"], rows, mids,
                caption="Classification streams: per-seed means with standard "
                        "deviations in parentheses, over the ten evaluation seeds.")


# ----------------------------------------------------------------------
# S4: classification detail and per-regime class distribution
# ----------------------------------------------------------------------
def build_s4():
    rows, mids = [], set()
    for b in ["email", "insects"]:
        cls = _per_block_classification(b)
        if not cls:
            continue
        if rows:
            mids.add(len(rows))
        for j, meth in enumerate(sorted(cls, key=order_key)):
            d = cls[meth]
            rows.append([BENCH_LABEL[b] if j == 0 else "", esc(meth)] +
                        [f"{num(d[k][0])} ({num(d[k][1])})"
                         for k in ["precision", "recall", "f1",
                                   "balanced_accuracy", "accuracy"]])
    write_table("s4_classification.tex", "ll" + "r" * 5,
                ["Benchmark", "Method", "Precision", "Recall", "$F_1$",
                 "Bal.\\ acc.", "Accuracy"], rows, mids,
                size=r"\scriptsize",
                colsep="2pt",
                caption="Breakdown of the classification metrics, aggregated exactly as in the main text: computed on each test block, averaged over blocks within a seed, then averaged over the ten seeds. Blocks straddling a drift point are excluded. Email uses the positive class; INSECTS uses the macro average over the six classes. Standard deviations across seeds in parentheses. The $F_1$ column reproduces Table~5 of the main text.")

    rows, mids = [], set()
    for b in ["email", "insects"]:
        rec = read_csv(os.path.join(OUTPUTS, b, "class_distribution",
                                    "class_distribution.csv"))
        rec = [r for r in rec if r["regime"] != "all"]
        if not rec:
            continue
        if rows:
            mids.add(len(rows))
        for j, r in enumerate(rec):
            frac = [f"{float(r[k]):.3f}" for k in r
                    if k.startswith("class_") and k.endswith("_frac") and r[k]]
            rows.append([BENCH_LABEL[b] if j == 0 else "", r["regime"],
                         f"{r['start']}--{r['end']}", r["n_samples"],
                         esc(r["region"]), ", ".join(frac)])
    write_table("s4b_class_distribution.tex", "llrrll",
                ["Benchmark", "Regime", "Samples", "$n$", "Window",
                 "Class fractions"], rows, mids,
                caption="Class distribution within each regime. On the email stream the imposed label reversals make the positive share alternate between $1/3$ and $2/3$; on INSECTS every regime is exactly uniform, so the drift there is a change in $p(y \\mid x)$ alone.")


# ----------------------------------------------------------------------
# S6: weight diagnostics
# ----------------------------------------------------------------------
DIAG = [("ess_over_N", "ESS$/N$"), ("weight_entropy", "Entropy"),
        ("max_weight", "Max weight"), ("unique_ancestor_rate_all", "Unique"),
        ("resample_rate", "Resample rate"), ("particle_spread", "Spread")]


def build_s6():
    rows, mids = [], set()
    for b in BENCH:
        rec = [r for r in read_csv(os.path.join(OUTPUTS, b, "summary_all.csv"))
               if r["experiment"] == "main" and r.get("kind") == "diagnostic"]
        if not rec:
            continue
        table = defaultdict(dict)
        for r in rec:
            table[r["method"]][r["metric"]] = (r["mean"], r["std"])
        if rows:
            mids.add(len(rows))
        for j, meth in enumerate(sorted(table, key=order_key)):
            cells = []
            for k, _ in DIAG:
                v = table[meth].get(k)
                cells.append("---" if v is None else f"{num(v[0])} ({num(v[1])})")
            rows.append([BENCH_LABEL[b] if j == 0 else "", esc(meth)] + cells)
    write_table("s6_weight_diagnostics.tex", "ll" + "r" * len(DIAG),
                ["Benchmark", "Method"] + [t for _, t in DIAG], rows, mids,
                size=r"\scriptsize",
                caption="Weight and ensemble diagnostics on the report window at $N=100$, averaged over ten seeds with standard deviations in parentheses. ``Unique'' is the fraction of distinct ancestors after resampling and ``Spread'' the trace of the particle covariance.")


# ----------------------------------------------------------------------
# S6b: numerical safeguards (conditioning, jitter, non-finite weights, clips)
# ----------------------------------------------------------------------
SAFE = [("cond_M_mean", "Cond.\\ mean"), ("cond_M_max", "Cond.\\ max"),
        ("jitter_count", "Jitter"), ("logcorr_nonfinite_count", "Non-finite"),
        ("rho_clip_count", "Clipped")]


def build_s6b():
    rows, mids = [], set()
    for b in BENCH:
        runs = os.path.join(OUTPUTS, b, "main", "runs")
        per = defaultdict(lambda: defaultdict(list))
        for d in sorted(glob.glob(os.path.join(runs, "*_seed*"))):
            meth = DIR2METHOD.get(os.path.basename(d).rsplit("_seed", 1)[0])
            fd = os.path.join(d, "diagnostics.npz")
            fm = os.path.join(d, "metrics.npz")
            if meth not in ("WSPF-A", "WSPF-B") or not os.path.exists(fd):
                continue
            zd, zm = np.load(fd), np.load(fm)
            mask = zm["report_mask"].astype(bool) & ~zm["straddle_mask"].astype(bool)
            for k, _ in SAFE:
                if k not in zd:
                    continue
                v = np.asarray(zd[k], dtype=float)[mask]
                v = v[np.isfinite(v)]
                if v.size:
                    # Conditioning is averaged; the counters are summed over
                    # the reporting window.
                    per[meth][k].append(float(v.mean()) if k.startswith("cond")
                                        else float(v.sum()))
        if not per:
            continue
        if rows:
            mids.add(len(rows))
        for j, meth in enumerate(sorted(per, key=order_key)):
            cells = []
            for k, _ in SAFE:
                v = per[meth].get(k)
                cells.append("---" if not v else
                             f"{num(np.mean(v), '{:.3g}')} ({num(np.std(v, ddof=1), '{:.2g}')})")
            rows.append([BENCH_LABEL[b] if j == 0 else "", esc(meth)] + cells)
    write_table("s6b_safeguards.tex", "ll" + "r" * len(SAFE),
                ["Benchmark", "Method"] + [t for _, t in SAFE], rows, mids,
                size=r"\scriptsize",
                caption="Numerical safeguards over the report window, averaged "
                        "over the ten seeds with standard deviations in "
                        "parentheses. ``Cond.'' is the condition number of the "
                        "$B \\times B$ system solved in Method~A, averaged and "
                        "maximized over particles within a step. ``Jitter'' "
                        "counts the particles whose Cholesky factorization "
                        "failed and needed a diagonal jitter, ``Non-finite'' the "
                        "non-finite log-corrections, and ``Clipped'' the "
                        "activations of the Method~B safeguard "
                        "$\\rho \\le 0.999$; all three are summed over the report "
                        "window. Method~B forms no $B \\times B$ system, so its "
                        "condition-number entries are empty.")


# ----------------------------------------------------------------------
# S7: sigma_cd sweep
# ----------------------------------------------------------------------
QCD = [("mse_mean", "Score"), ("ess_over_N", "ESS$/N$"),
       ("rho_q50", r"$\rho_{50}$"), ("rho_q90", r"$\rho_{90}$"),
       ("rho_q99", r"$\rho_{99}$"), ("rho_max", r"$\rho_{\max}$"),
       ("P_rho_gt_0.9", "$P(\\rho>0.9)$"), ("P_rho_gt_0.99", "$P(\\rho>0.99)$"),
       ("wspf_b_actual_clip_rate", "Clip rate"), ("nonfinite", "Non-finite")]


def build_s7():
    rows, mids = [], set()
    for b in BENCH:
        rec = read_csv(os.path.join(OUTPUTS, b, "qcd_sweep", "metrics.csv"))
        rec = [r for r in rec if r.get("method") in ("PF", "WSPF-A", "WSPF-B")]
        if not rec:
            continue
        rec.sort(key=lambda r: (order_key(r["method"]), float(r["sigma_cd"])))
        if rows:
            mids.add(len(rows))
        last = None
        for r in rec:
            lab = BENCH_LABEL[b] if last is None and not rows else ""
            meth = esc(r["method"]) if r["method"] != last else ""
            last = r["method"]
            rows.append([lab, meth, num(r["sigma_cd"], "{:g}")] +
                        [num(r.get(k)) for k, _ in QCD])
    write_table("s7_qcd_sweep.tex", "lll" + "r" * len(QCD),
                ["Benchmark", "Method", r"$\sigma_{\mathrm{cd}}$"] +
                [t for _, t in QCD], rows, mids,
                size=r"\scriptsize",
                caption="Sensitivity to the concept-drift scale $\\sigma_{\\mathrm{cd}}$. ``Score'' is test MSE for the regression benchmark and $F_1$ for the classification streams. $\\rho_q$ are quantiles of the per-particle $\\rho_t^{(i)}$ over the report window, ``Clip rate'' the fraction of steps at which the Method~B safeguard $\\rho \\le 0.999$ binds, and ``Non-finite'' the count of non-finite weights.")


# ----------------------------------------------------------------------
# S8: gradient-noise analysis
# ----------------------------------------------------------------------
GN = [("mean_abs_skew", "$|$skew$|$"), ("mean_kurtosis", "Kurtosis"),
      ("frac_reject_normal", "Reject rate"), ("median_normal_p", "Median $p$"),
      ("fisher_normal_p", "Fisher $p$"), ("mean_maha2", r"$\bar D^2$"),
      ("expected_maha2", r"$\mathbb{E}D^2$"), ("maha_ks_p", "KS $p$"),
      ("cond_number", "Cond."), ("effective_rank", "Eff.\\ rank"),
      ("top_eig_ratio", "Top eig.")]


def build_s8():
    rec = read_csv(os.path.join(OUTPUTS, "regression", "grad_noise",
                                "gradient_noise.csv"))
    rec.sort(key=lambda r: (int(float(r["batch_size"])), r["phase"]))
    rows, mids, last = [], set(), None
    for r in rec:
        if last is not None and r["batch_size"] != last:
            mids.add(len(rows))
        b = r["batch_size"] if r["batch_size"] != last else ""
        last = r["batch_size"]
        rows.append([b, esc(r["phase"])] + [num(r.get(k)) for k, _ in GN])
    write_table("s8_gradient_noise.tex", "ll" + "r" * len(GN),
                ["$B$", "Phase"] + [t for _, t in GN], rows, mids,
                size=r"\scriptsize",
                caption="Empirical assessment of the Gaussian approximation to mini-batch gradient noise on the regression benchmark, by batch size and phase relative to a concept switch. $\\bar D^2$ is the mean squared Mahalanobis distance against its expectation $\\mathbb{E}D^2 = d$, and ``KS $p$'' compares its distribution with $\\chi^2(d)$. Each cell uses 3000 independent mini-batches, with the population gradient estimated from about $10^5$ samples.")


# ----------------------------------------------------------------------
# S9: compute cost
# ----------------------------------------------------------------------
CC = [("t_step_ms", "Step (ms)"), ("t_correction_ms", "Correction (ms)"),
      ("sample_grad_evals_per_step", "Grad.\\ evals"),
      ("peak_python_traced_mem_MB", "Peak (MB)"),
      ("static_ema_MB", "EMA"), ("static_particles_MB", "Particles"),
      ("static_deviations_MB", "Dev."), ("static_BxB_MB", "$B\\times B$"),
      ("static_solve_MB", "Solve"), ("static_total_MB", "Static total")]


def build_s9():
    rows, mids = [], set()
    for b in BENCH:
        rec = [r for r in read_csv(os.path.join(OUTPUTS, b, "compute_cost", "metrics.csv"))
               if r.get("N") in ("", "100", "100.0")]
        if not rec:
            rec = read_csv(os.path.join(OUTPUTS, b, "compute_cost", "metrics.csv"))
        if not rec:
            continue
        rec.sort(key=lambda r: order_key(r["method"]))
        if rows:
            mids.add(len(rows))
        for j, r in enumerate(rec):
            rows.append([BENCH_LABEL[b] if j == 0 else "", esc(r["method"])] +
                        [num(r.get(k), "{:.3g}") for k, _ in CC])
    write_table("s9_compute_cost.tex", "ll" + "r" * len(CC),
                ["Benchmark", "Method"] + [t for _, t in CC], rows, mids,
                size=r"\scriptsize",
                caption="Computational cost per online update at $N=100$: wall-clock time per step, the part of it spent in the correction, per-sample gradient evaluations, peak traced memory, and the static memory breakdown in MB.")

    rec = read_csv(os.path.join(OUTPUTS, "regression", "compute_cost", "dim_sweep.csv"))
    if rec:
        keys = [k for k in rec[0] if k not in ("method",)]
        rows = [[esc(r["method"])] + [num(r[k], "{:.3g}") for k in keys] for r in rec]
        rows.sort(key=lambda r: order_key(r[0].replace("\\_", "_")))
        write_table("s9b_dim_sweep.tex", "l" + "r" * len(keys),
                    ["Method"] + [esc(k) for k in keys], rows,
                    caption="Cost as the parameter dimension is varied on the regression benchmark.")


# ----------------------------------------------------------------------
# S5: copy the reliability diagrams
# ----------------------------------------------------------------------
def build_s5():
    n = 0
    for b in ["email", "insects"]:
        src = os.path.join(OUTPUTS, b, "calibration")
        for f in sorted(glob.glob(os.path.join(src, "reliability_*.png"))):
            dst = os.path.join(SUPP, f"{b}_" + os.path.basename(f))
            shutil.copyfile(f, dst)
            n += 1
    print(f"  copied {n} reliability figures")


# ----------------------------------------------------------------------
def main():
    os.makedirs(SUPP, exist_ok=True)
    print(f"building supplementary tables into {os.path.relpath(SUPP, ROOT)}/")
    build_s1(); build_s2(); build_s3(); build_s3b(); build_s3c(); build_s4()
    build_s5(); build_s6(); build_s6b(); build_s7(); build_s8(); build_s9()

    # Self-check against Table 5 of the main text, confirming that the
    # aggregation convention still matches.
    cls = _per_block_classification("email")
    ref = {"PF": 0.7937, "WSPF-A": 0.7704, "WSPF-B": 0.8158,
           "SGD": 0.7883, "PH-SGD": 0.7750, "Window-SGD": 0.7920}
    bad = [m for m, v in ref.items()
           if abs(cls[m]["f1"][0] - v) > 5e-5]
    print("  self-check vs. Table 5 F1:",
          "OK" if not bad else f"MISMATCH {bad}")


if __name__ == "__main__":
    main()
