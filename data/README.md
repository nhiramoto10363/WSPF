# Obtaining the datasets

This directory ships empty. The two real-world streams are not redistributed
here; please obtain them from their original sources. (The regression
benchmark is synthetic and needs no download.)

Place the files in this directory under these names:

| File | Benchmark | Source | License |
|---|---|---|---|
| `email_data.arff` | Email (elist) | Katakis, Tsoumakas & Vlahavas (2010) | per the distributor's terms |
| `INSECTS-abrupt_balanced_norm.csv` | INSECTS | Souza, Reis, Maletzke & Batista (2020), USP DS repository | CC BY 4.0 |

## 1. Email (elist)

I. Katakis, G. Tsoumakas, I. Vlahavas.
*Tracking recurring contexts using ensemble classifiers: an application to
email filtering.* Knowledge and Information Systems, 22(3):371-391, 2010.
doi:10.1007/s10115-009-0206-2

Section 6 of that paper points to <http://mlkd.csd.auth.gr/concept_drift.html>
(Datasets 3), which distributes the stream in ARFF format with a boolean
bag-of-words representation. If that page is unreachable, contact the authors
or use a concept-drift benchmark collection that includes the same stream;
the verification step below confirms you have the stream the paper used.

Expected content: 1,500 samples, 913 binary bag-of-words attributes plus a
label. Concept switches occur at samples 300, 600, 900 and 1200, and the
positive-class share alternates 1/3, 2/3, 1/3, 2/3, 1/3 across the five
periods (Table 4 of the paper).

## 2. INSECTS (abrupt, balanced)

V. M. A. Souza, D. M. Reis, A. G. Maletzke, G. E. A. P. A. Batista.
*Challenges in Benchmarking Stream Learning Algorithms with Real-world Data.*
Data Mining and Knowledge Discovery, 34:1805-1858, 2020.
doi:10.1007/s10618-020-00698-5

Download the **normalized abrupt (balanced) variant**,
`INSECTS-abrupt_balanced_norm.csv`, from the USP DS repository at
<https://sites.google.com/view/uspdsrepository>. The repository is published
under CC BY 4.0 and asks that both the paper and the repository be credited.

Expected content: 52,848 samples, 33 real-valued features and 6 classes, with
known change points at samples 14352, 19500, 33240, 38682 and 39510. **Using
the wrong variant invalidates every switch-aligned analysis**, so run the
verification below.

## 3. Placing and verifying the files

```bash
python scripts/prepare_data.py \
    --email-from-file   /path/to/email_data.arff \
    --insects-from-file /path/to/INSECTS-abrupt_balanced_norm.csv
```

Pass `--email-url` / `--insects-url` to download instead, or `--check` to
verify files already present in this directory.

Verification compares the SHA-256 against the version used for the paper and
then checks the stream structure: sample count, feature dimension and label
composition (per-period positive rate for elist; class count and change points
for INSECTS). A SHA-256 mismatch with a passing structure check means the
content matches up to details such as line endings.

## 4. Preprocessing

Dimensionality reduction and standardization are deliberately not applied
here. To keep information from the reporting window out of feature
extraction, both are fit **on an initial segment only** at experiment time and
then applied to the whole stream:

- elist: centering plus truncated SVD to 50 components, fit on samples
  `[0, 600)` (`src/benchmarks/loaders/email_loader.py`).
- INSECTS: per-feature standardization, fit on samples `[0, 20000)`
  (`src/benchmarks/loaders/insects_loader.py`).

Both are applied automatically when the benchmark is constructed, so no
manual preprocessing step is required.
