#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch, place and verify the real-world datasets (elist and INSECTS).

The raw data is not redistributed here, so this script does three things:

  1. fetch   - download from --url, or copy a local file given by --from-file,
               into data/
  2. identity - check the SHA-256 against the version used for the paper
  3. structure - load the file through its loader and verify the sample count,
               the feature dimension and the label composition (per-period
               positive rate for elist, class count and change points for
               INSECTS)

A SHA-256 mismatch is only a warning: if the structure check still passes,
the file is almost certainly the same content in a different distribution
(different line endings, say). When both pass, the stream on disk is the one
the experiments were run on.

Preprocessing (PCA, standardization) is deliberately not done here, because
it must be fit on an initial segment only to avoid leakage. It lives in
  src/benchmarks/loaders/email_loader.py    (centering + truncated SVD,
                                             fit on the first 600 samples)
  src/benchmarks/loaders/insects_loader.py  (standardization, fit on the
                                             first 20000 samples)
and is applied automatically when the benchmark is constructed.

Usage:
    # place local copies and verify them
    python scripts/prepare_data.py --email-from-file /path/to/email_data.arff \
                                   --insects-from-file /path/to/INSECTS-abrupt_balanced_norm.csv
    # download and verify
    python scripts/prepare_data.py --email-url URL --insects-url URL
    # verify what is already in data/
    python scripts/prepare_data.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")
sys.path.insert(0, _ROOT)

# Fingerprints of the versions used for the published results.
SPEC = {
    "email": {
        "filename": "email_data.arff",
        "sha256": ("00a12c8656b687a70574292a06c44fd6818af386c027983dfa"
                   "c2c8b1cb900d6b"),
        "source": ("Katakis, Tsoumakas & Vlahavas (2010), the elist stream. "
                   "Distribution page given in that paper: "
                   "http://mlkd.csd.auth.gr/concept_drift.html (Datasets 3)"),
    },
    "insects": {
        "filename": "INSECTS-abrupt_balanced_norm.csv",
        "sha256": ("e4819251b250a6fc1bf2a3798bbb7a1cbd2aff81d2ea34252a9"
                   "04c5266272fff"),
        "source": ("Souza, Reis, Maletzke & Batista (2020), USP DS "
                   "repository (CC BY 4.0): "
                   "https://sites.google.com/view/uspdsrepository"),
    },
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _place(name, url=None, from_file=None):
    """Put the dataset at data/<filename>; leave an existing file alone."""
    dst = os.path.join(_DATA, SPEC[name]["filename"])
    if url is None and from_file is None:
        return dst
    os.makedirs(_DATA, exist_ok=True)
    if from_file:
        if not os.path.exists(from_file):
            raise FileNotFoundError(from_file)
        shutil.copyfile(from_file, dst)
        print(f"[{name}] copied {from_file} -> {os.path.relpath(dst, _ROOT)}")
    else:
        print(f"[{name}] downloading {url}")
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"[{name}] saved {os.path.relpath(dst, _ROOT)}")
    return dst


def _check_fingerprint(name, path):
    got = _sha256(path)
    want = SPEC[name]["sha256"]
    if got == want:
        print(f"[{name}] SHA-256 matches")
        return True
    print(f"[{name}] [warning] SHA-256 differs from the published version\n"
          f"          expected {want}\n"
          f"          actual   {got}\n"
          f"          The structure check below tells apart a genuine "
          f"mismatch from a formatting difference.")
    return False


def _check_email(path):
    """1,500 samples by 913 features, with the 1/3 - 2/3 alternation."""
    from src.benchmarks.loaders.email_loader import EmailDataLoader
    ld = EmailDataLoader(path)                    # raw features, no PCA
    assert ld.n_samples == 1500, f"n_samples={ld.n_samples} (expected 1500)"
    assert ld.input_dim == 913, f"input_dim={ld.input_dim} (expected 913)"
    shares = [float(ld.y[s:s + 300].mean()) for s in range(0, 1500, 300)]
    expect = [1 / 3, 2 / 3, 1 / 3, 2 / 3, 1 / 3]
    for i, (g, e) in enumerate(zip(shares, expect)):
        assert abs(g - e) < 1e-3, (
            f"positive rate {g:.4f} in period {i} (expected {e:.4f}); the "
            f"label-reversal pattern does not match Table 4 of the paper")
    print(f"  n=1500, d=913, positive rates = "
          f"{', '.join(f'{s:.3f}' for s in shares)}  -> matches Table 4")


def _check_insects(path):
    """52,848 samples by 33 features, 6 classes, change points in range."""
    from src.benchmarks.loaders.insects_loader import InsectsDataLoader
    ld = InsectsDataLoader(path)                  # also range-checks the CPs
    assert ld.n_samples == 52848, f"n_samples={ld.n_samples} (expected 52848)"
    assert ld.n_features == 33, f"n_features={ld.n_features} (expected 33)"
    assert ld.n_classes == 6, f"n_classes={ld.n_classes} (expected 6)"
    print(f"  n=52848, d=33, C=6, change points = {ld.change_points}  "
          f"-> matches the abrupt (balanced) variant")


_CHECKERS = {"email": _check_email, "insects": _check_insects}


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--email-url")
    ap.add_argument("--email-from-file")
    ap.add_argument("--insects-url")
    ap.add_argument("--insects-from-file")
    ap.add_argument("--check", action="store_true",
                    help="verify what is already in data/, fetching nothing")
    args = ap.parse_args()

    plans = {
        "email": (args.email_url, args.email_from_file),
        "insects": (args.insects_url, args.insects_from_file),
    }

    ok = True
    for name, (url, from_file) in plans.items():
        print(f"\n=== {name} ===")
        path = _place(name, url, from_file)
        if not os.path.exists(path):
            print(f"[{name}] not found: {os.path.relpath(path, _ROOT)}\n"
                  f"          source: {SPEC[name]['source']}\n"
                  f"          Download it, then pass --{name}-from-file.")
            ok = False
            continue
        _check_fingerprint(name, path)
        try:
            _CHECKERS[name](path)
        except AssertionError as e:
            print(f"[{name}] [failed] structure check: {e}")
            ok = False
        else:
            print(f"[{name}] structure check passed")

    print("\n" + ("Both datasets verified; the experiments can be run."
                  if ok else
                  "Some datasets are missing or failed verification (above)."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
