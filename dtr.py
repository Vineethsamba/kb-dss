#!/usr/bin/env python3
"""KB-DSS: knowledge-based decision support for digital transformation readiness.

Single reproduction script for the IEEE manuscript. It generates the synthetic
benchmark, runs every experiment, and writes Fig. 1, Fig. 2, Table 1, Table 2
and a metrics file with fixed random seeds.

Usage
-----
    python dtr.py
    python dtr.py --data firms.csv

CSV schema for --data
---------------------
Required columns (values in [0, 1] unless noted):
    data_maturity, analytics_capability, ai_adoption, cloud_adoption,
    data_governance, leadership_commitment, digital_skills, rd_intensity,
    innovation_output, entrepreneurial_orientation, legacy_debt,
    regulatory_intensity, log_size, sector, y
sector is an integer in {0,1,2,3,4} (manufacturing, services, finance,
healthcare, other). y is 0=Low, 1=Medium, 2=High, or the strings
Low/Medium/High. log_size is standardized log firm size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
INDICATORS = [
    "data_maturity",
    "analytics_capability",
    "ai_adoption",
    "cloud_adoption",
    "data_governance",
    "leadership_commitment",
    "digital_skills",
    "rd_intensity",
    "innovation_output",
    "entrepreneurial_orientation",
    "legacy_debt",
    "regulatory_intensity",
]
I = {name: j for j, name in enumerate(INDICATORS)}
SECTORS = ["manufacturing", "services", "finance", "healthcare", "other"]
CLASS_NAMES = np.array(["Low", "Medium", "High"])
N_FIRMS = 2000
SEED = 42

PILLAR_SPEC = [
    ("Data", [(I["data_maturity"], 0.5), (I["analytics_capability"], 0.5)]),
    ("Technology", [(I["ai_adoption"], 0.35), (I["cloud_adoption"], 0.35), (I["legacy_debt"], -0.30)]),
    ("Organization", [(I["leadership_commitment"], 0.5), (I["digital_skills"], 0.5)]),
    ("Governance", [(I["data_governance"], 1.0)]),
    ("Innovation", [(I["rd_intensity"], 0.30), (I["innovation_output"], 0.35), (I["entrepreneurial_orientation"], 0.35)]),
]
RULES = [
    ("R1_data_ai_synergy", +1, [(I["data_maturity"], False), (I["ai_adoption"], False)]),
    ("R2_governance_gap", -1, [(I["ai_adoption"], False), (I["data_governance"], True), (I["regulatory_intensity"], False)]),
    ("R3_legacy_barrier", -1, [(I["legacy_debt"], False), (I["cloud_adoption"], True)]),
    ("R4_sponsor_capability", +1, [(I["leadership_commitment"], False), (I["digital_skills"], False)]),
    ("R5_innovation_engine", +1, [(I["rd_intensity"], False), (I["entrepreneurial_orientation"], False)]),
    ("R6_analytics_without_base", -1, [(I["analytics_capability"], False), (I["data_maturity"], True)]),
]
OMEGA = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
LAMBDA = 0.35
HGB_KW = dict(max_iter=150, learning_rate=0.06, max_depth=3, l2_regularization=1.0)
KNOWLEDGE_NAMES = (
    [f"P_{name}" for name, _ in PILLAR_SPEC]
    + [name for name, _, _ in RULES]
    + ["KB_index"]
)


def self_test() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((32, 12))
    p = pillar_scores(x)
    m = rule_activations(x)
    e = expert_index(p, m)
    assert p.shape == (32, 5) and np.all((p >= 0) & (p <= 1))
    assert m.shape == (32, 6) and np.all((m >= 0) & (m <= 1))
    assert np.all(np.isfinite(e))
    k = kappa(x)
    assert k.shape == (32, 12)
    x0 = apply_fidelity(x, 0.0, np.random.default_rng(1))
    assert not np.allclose(x0, x)
    x1 = apply_fidelity(x, 1.0, np.random.default_rng(1))
    assert np.allclose(x1, x)


def pillar_scores(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    p = np.zeros((n, len(PILLAR_SPEC)))
    for k, (_, terms) in enumerate(PILLAR_SPEC):
        num = np.zeros(n)
        den = 0.0
        for j, w in terms:
            xt = x[:, j] if w > 0 else 1.0 - x[:, j]
            num += abs(w) * xt
            den += abs(w)
        p[:, k] = num / den
    return p


def rule_activations(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    mu = np.ones((n, len(RULES)))
    for r, (_, _, lits) in enumerate(RULES):
        act = np.ones(n)
        for j, complement in lits:
            lit = 1.0 - x[:, j] if complement else x[:, j]
            act = np.minimum(act, lit)
        mu[:, r] = act
    return mu


def expert_index(p: np.ndarray, mu: np.ndarray, signs=None, omega=None) -> np.ndarray:
    signs = np.array([s for _, s, _ in RULES], dtype=float) if signs is None else np.asarray(signs, dtype=float)
    omega = OMEGA if omega is None else np.asarray(omega, dtype=float)
    return p @ omega + (LAMBDA / 6.0) * (mu * signs).sum(axis=1)


def pillar_scores_spec(x: np.ndarray, spec) -> np.ndarray:
    n = x.shape[0]
    p = np.zeros((n, len(spec)))
    for k, (_, terms) in enumerate(spec):
        num = np.zeros(n)
        den = 0.0
        for j, w in terms:
            if abs(w) < 1e-12:
                continue
            xt = x[:, j] if w > 0 else 1.0 - x[:, j]
            num += abs(w) * xt
            den += abs(w)
        p[:, k] = num / den if den > 0 else 0.5
    return p


def kappa(x: np.ndarray, layer: dict | None = None) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    spec = PILLAR_SPEC if layer is None else layer.get("spec", PILLAR_SPEC)
    signs = None if layer is None else layer.get("signs")
    omega = None if layer is None else layer.get("omega")
    drop = set() if layer is None else set(layer.get("drop", ()))
    p = pillar_scores_spec(x, spec)
    mu = rule_activations(x)
    if drop:
        mu = mu.copy()
        mu[:, list(drop)] = 0.0
    e = expert_index(p, mu, signs=signs, omega=omega)
    return np.column_stack([p, mu, e])


def corrupt_layer(kind: str, rng: np.random.Generator) -> dict:
    """Alternative knowledge-failure modes beyond indicator permutation."""
    signs = np.array([s for _, s, _ in RULES], dtype=float)
    spec = [(name, list(terms)) for name, terms in PILLAR_SPEC]
    omega = OMEGA.copy()
    drop: tuple[int, ...] = ()
    if kind == "flip_all_signs":
        signs = -signs
    elif kind == "flip_three_signs":
        idx = rng.choice(6, size=3, replace=False)
        signs[idx] *= -1
    elif kind == "weight_noise":
        noisy = []
        for name, terms in spec:
            noisy.append(
                (name, [(j, float(np.clip(w + rng.normal(0.0, 0.25), -1.0, 1.0))) for j, w in terms])
            )
        spec = noisy
        omega = np.abs(omega + rng.normal(0.0, 0.08, size=5))
        omega = omega / omega.sum()
    elif kind == "drop_aligned_rules":
        drop = (0, 1, 2, 4)  # R1, R2, R3, R5 appear in the generator
    elif kind == "drop_extra_rules":
        drop = (3, 5)  # R4, R6 have no generator counterpart
    elif kind != "correct":
        raise ValueError(kind)
    return {"spec": spec, "signs": signs, "omega": omega, "drop": drop}


def apply_fidelity(x: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    n_wrong = int(round((1.0 - phi) * 12))
    if n_wrong <= 0:
        return x.copy()
    idx = np.sort(rng.choice(12, size=min(n_wrong, 12), replace=False))
    out = x.copy()
    out[:, idx] = np.roll(out[:, idx], shift=1, axis=1)
    return out


def generate_benchmark(n: int = N_FIRMS, seed: int = SEED):
    rng = np.random.default_rng(seed)
    sector = rng.choice(5, size=n, p=[0.24, 0.24, 0.16, 0.16, 0.20])
    raw_size = rng.lognormal(mean=4.2, sigma=0.9, size=n)
    log_size = np.log(raw_size)
    log_size = (log_size - log_size.mean()) / log_size.std(ddof=0)

    culture = rng.beta(2.2, 2.2, size=n)
    mix = 0.38
    x = np.empty((n, 12))
    for j in range(12):
        x[:, j] = mix * culture + (1.0 - mix) * rng.beta(2.0, 2.0, size=n)

    x[:, I["legacy_debt"]] = np.clip(
        0.55 * x[:, I["legacy_debt"]] + 0.45 * (1.0 - x[:, I["cloud_adoption"]]), 0, 1
    )
    boost = np.isin(sector, [2, 3]).astype(float)
    x[:, I["regulatory_intensity"]] = np.clip(
        x[:, I["regulatory_intensity"]] + 0.22 * boost, 0, 1
    )
    x[:, I["data_governance"]] = np.clip(
        0.65 * x[:, I["data_governance"]] + 0.35 * x[:, I["regulatory_intensity"]], 0, 1
    )

    # Generator weights differ from the expert pillar weights. The systematic
    # component is scaled so that Gaussian noise with SD 0.55 leaves a
    # learnable three-class problem (Macro-F1 near the manuscript range).
    w = np.array([0.13, 0.07, 0.11, 0.12, 0.08, 0.11, 0.09, 0.06, 0.10, 0.08, -0.15, -0.06])
    latent = x @ w
    latent += 0.20 * (x[:, I["data_maturity"]] * x[:, I["ai_adoption"]])
    latent += -0.16 * (
        x[:, I["ai_adoption"]]
        * (1.0 - x[:, I["data_governance"]])
        * x[:, I["regulatory_intensity"]]
    )
    latent += -0.18 * (x[:, I["legacy_debt"]] * (1.0 - x[:, I["cloud_adoption"]]))
    latent += 0.14 * (x[:, I["rd_intensity"]] * x[:, I["entrepreneurial_orientation"]])
    latent += 0.11 * (x[:, I["cloud_adoption"]] * log_size)
    latent += np.array([0.00, 0.05, -0.07, -0.06, 0.04])[sector]
    scale = 1.55 / (float(latent.std()) + 1e-12)
    latent = scale * latent + rng.normal(0.0, 0.55, size=n)

    t30, t70 = np.quantile(latent, [0.30, 0.70])
    y = np.where(latent <= t30, 0, np.where(latent <= t70, 1, 2)).astype(int)
    return x, log_size, sector, y, latent


def load_real(path: Path):
    df = pd.read_csv(path)
    missing = [c for c in INDICATORS + ["log_size", "sector", "y"] if c not in df.columns]
    if missing:
        raise SystemExit(f"--data file is missing columns: {missing}")
    x = df[INDICATORS].to_numpy(float)
    if np.any((x < 0) | (x > 1)):
        raise SystemExit("Indicator columns must lie in [0, 1].")
    log_size = df["log_size"].to_numpy(float)
    sector = df["sector"].to_numpy(int)
    y_raw = df["y"]
    if y_raw.dtype == object or str(y_raw.dtype).startswith("str"):
        mapping = {"low": 0, "medium": 1, "med": 1, "high": 2}
        y = y_raw.astype(str).str.lower().map(mapping).to_numpy()
        if np.any(pd.isna(y)):
            raise SystemExit("y must be Low/Medium/High or 0/1/2.")
        y = y.astype(int)
    else:
        y = y_raw.to_numpy(int)
    return x, log_size, sector, y


def raw_joint(x, log_size, sector) -> np.ndarray:
    return np.column_stack([x, log_size, sector.astype(float)])


def expert_predict(e_train, y_train, e_test) -> np.ndarray:
    priors = np.array([(y_train == c).mean() for c in range(3)])
    q1 = priors[0]
    q2 = priors[0] + priors[1]
    t1, t2 = np.quantile(e_train, [q1, q2])
    return np.where(e_test <= t1, 0, np.where(e_test <= t2, 1, 2)).astype(int)


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def make_hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, **HGB_KW)


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )


def make_lr() -> LogisticRegression:
    try:
        return LogisticRegression(max_iter=400, C=1.0)
    except TypeError:
        return LogisticRegression(max_iter=400, C=1.0, multi_class="auto")


def encode_lr(raw: np.ndarray, scaler: StandardScaler | None = None, fit: bool = False):
    # Column 13 is sector (0-4). All other columns are continuous, including
    # optional knowledge features appended after the 14 raw inputs.
    sector = raw[:, 13].astype(int)
    cont = np.delete(raw, 13, axis=1)
    oh = np.eye(5)[sector]
    if fit:
        scaler = StandardScaler()
        cont = scaler.fit_transform(cont)
        return np.hstack([cont, oh]), scaler
    cont = scaler.transform(cont)
    return np.hstack([cont, oh]), scaler


def fit_predict(name, x_tr, y_tr, x_te, seed: int):
    if name == "expert":
        e_tr, e_te = x_tr[:, -1], x_te[:, -1]
        pred = expert_predict(e_tr, y_tr, e_te)
        return pred, None
    if name in {"hgb", "kbdss"}:
        clf = make_hgb(seed)
        clf.fit(x_tr, y_tr)
        return clf.predict(x_te), clf
    if name == "rf":
        clf = make_rf(seed)
        clf.fit(x_tr, y_tr)
        return clf.predict(x_te), clf
    if name in {"lr", "kblr"}:
        Z_tr, scaler = encode_lr(x_tr, fit=True)
        Z_te, _ = encode_lr(x_te, scaler=scaler)
        clf = make_lr()
        clf.fit(Z_tr, y_tr)
        return clf.predict(Z_te), clf
    raise ValueError(name)


def nadeau_bengio(diffs: np.ndarray, n_train: int, n_test: int) -> dict:
    j = len(diffs)
    mean = diffs.mean()
    var = diffs.var(ddof=1)
    se = np.sqrt((1.0 / j + n_test / n_train) * var)
    tstat = mean / se if se > 0 else np.inf
    from math import erf, sqrt

    # two-sided p from Student-t with J-1 df via normal fallback is coarse;
    # use a t CDF approximation.
    df = j - 1
    z = abs(tstat)
    # Abramowitz-Stegun-ish t-to-p via regularized incomplete beta would be
    # heavier; scipy may not be present. Use a t-tail approximation.
    p = 2.0 * (1.0 - _t_cdf(z, df))
    ci = 1.96 * se
    return {
        "mean": float(mean),
        "ci_lo": float(mean - ci),
        "ci_hi": float(mean + ci),
        "p": float(p),
        "t": float(tstat),
    }


def _t_cdf(t: float, df: float) -> float:
    # Incomplete beta representation of Student's t CDF for t>0.
    x = df / (df + t * t)
    a = 0.5 * df
    b = 0.5
    return 1.0 - 0.5 * _betainc(a, b, x)


def _betainc(a, b, x):
    # Regularized incomplete beta via continued fraction (Lentz), x in (0,1).
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    from math import exp, lgamma, log

    def _cf(aa, bb, xx):
        qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
        c, d = 1.0, 1.0 - qab * xx / qap
        d = 1e30 if abs(d) < 1e-30 else 1.0 / d
        f = d
        for m in range(1, 200):
            m2 = 2 * m
            num = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + num * d
            c = 1.0 + num / c
            d = 1e30 if abs(d) < 1e-30 else d
            c = 1e30 if abs(c) < 1e-30 else c
            d = 1.0 / d
            f *= d * c
            num = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + num * d
            c = 1.0 + num / c
            d = 1e30 if abs(d) < 1e-30 else d
            c = 1e30 if abs(c) < 1e-30 else c
            d = 1.0 / d
            delta = d * c
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                break
        return f

    ln_front = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return exp(ln_front) * _cf(a, b, x) / a
    return 1.0 - exp(ln_front) * _cf(b, a, 1.0 - x) / b


def summarize_folds(rows):
    acc = np.array([r["accuracy"] for r in rows])
    f1 = np.array([r["macro_f1"] for r in rows])
    qwk = np.array([r["qwk"] for r in rows])
    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_sd": float(acc.std(ddof=1)),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_sd": float(f1.std(ddof=1)),
        "qwk_mean": float(qwk.mean()),
        "qwk_sd": float(qwk.std(ddof=1)),
        "f1_folds": f1.tolist(),
    }


def run_cv(x, log_size, sector, y, n_repeats=3, n_splits=5, label_noise=0.0, phi=1.0, components="full"):
    results = {k: [] for k in ["expert", "lr", "rf", "hgb", "kblr", "kbdss"]}
    rng_master = np.random.default_rng(SEED)
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + 100 * rep)
        for fold, (tr, te) in enumerate(skf.split(x, y)):
            y_tr = y[tr].copy()
            if label_noise > 0:
                n_flip = int(round(label_noise * len(y_tr)))
                idx = rng_master.choice(len(y_tr), size=n_flip, replace=False)
                y_tr[idx] = rng_master.integers(0, 3, size=n_flip)
            rng_phi = np.random.default_rng(SEED + 17 * rep + fold + int(1000 * phi))
            x_tr_k = apply_fidelity(x[tr], phi, rng_phi) if components != "raw" else x[tr]
            x_te_k = apply_fidelity(x[te], phi, rng_phi) if components != "raw" else x[te]
            k_tr, k_te = kappa(x_tr_k), kappa(x_te_k)
            if components == "pillars":
                k_tr, k_te = k_tr[:, :5], k_te[:, :5]
            elif components == "rules":
                k_tr, k_te = k_tr[:, 5:11], k_te[:, 5:11]
            elif components == "index":
                k_tr, k_te = k_tr[:, 11:12], k_te[:, 11:12]
            raw_tr = raw_joint(x[tr], log_size[tr], sector[tr])
            raw_te = raw_joint(x[te], log_size[te], sector[te])
            full_tr = np.hstack([raw_tr, k_tr]) if components != "raw" else raw_tr
            full_te = np.hstack([raw_te, k_te]) if components != "raw" else raw_te
            seed = SEED + 3 * rep + fold
            if components == "full" and label_noise == 0 and phi == 1.0:
                for name, tr_x, te_x in [
                    ("expert", full_tr, full_te),
                    ("lr", raw_tr, raw_te),
                    ("rf", raw_tr, raw_te),
                    ("hgb", raw_tr, raw_te),
                    ("kblr", full_tr, full_te),
                    ("kbdss", full_tr, full_te),
                ]:
                    pred, _ = fit_predict(name, tr_x, y_tr, te_x, seed)
                    results[name].append(metrics(y[te], pred))
            else:
                pred, _ = fit_predict("kbdss" if components != "raw" else "hgb", full_tr, y_tr, full_te, seed)
                results["kbdss"].append(metrics(y[te], pred))
                if components == "raw" or label_noise > 0:
                    pred_h, _ = fit_predict("hgb", raw_tr, y_tr, raw_te, seed)
                    results["hgb"].append(metrics(y[te], pred_h))
    return results


def fmt_row(name, s, gain=None):
    base = (
        f"{name:42s}  "
        f"{s['accuracy_mean']:.3f} ± {s['accuracy_sd']:.3f}  "
        f"{s['macro_f1_mean']:.3f} ± {s['macro_f1_sd']:.3f}  "
        f"{s['qwk_mean']:.3f} ± {s['qwk_sd']:.3f}"
    )
    if gain is None:
        return base
    return base + f"  {gain['mean']:+.3f} [{gain['ci_lo']:+.3f}, {gain['ci_hi']:+.3f}]  p={gain['p']:.3f}"


def draw_fig1(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.8, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 0.7, 2.5, 2.6, "#E8F1FB", "#2F6FED", "1. Enterprise data",
         "12 innovation indicators\n+ firm size + sector"),
        (3.2, 0.7, 2.7, 2.6, "#FFF4E5", "#E07A2F", "2. Knowledge layer",
         "5 expert pillars\n6 fuzzy IF–THEN rules\nExpert readiness index"),
        (6.3, 0.7, 2.5, 2.6, "#E7F6EE", "#2E8B57", "3. Hybrid learner",
         "Raw inputs\n+ knowledge features\nGradient boosting"),
        (9.2, 0.7, 2.5, 2.6, "#F8E8F2", "#A64D79", "4. Decision support",
         "Readiness Low / Med / High\nRule trace\nWeakest pillars + actions"),
    ]
    for x, y, w, h, fc, ec, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    facecolor=fc, edgecolor=ec, linewidth=2))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", fontsize=8, color="#333333")
    for x in (2.7, 5.9, 8.8):
        ax.annotate("", xy=(x + 0.5, 2.0), xytext=(x, 2.0),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.4))
    ax.annotate("", xy=(9.2, 0.45), xytext=(3.2, 0.45),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0, ls="--"))
    ax.text(6.2, 0.18, "knowledge features (pillar scores, rule activations, index) are appended to the raw inputs",
            ha="center", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_fig2(path: Path, curve, heat, imp_names, imp_vals, is_knowledge) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    sizes = curve["sizes"]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7), gridspec_kw={"width_ratios": [1.15, 1.2, 1.0]})

    ax = axes[0]
    styles = {
        "expert": ("#888888", "s", "Expert index (KB only)"),
        "lr": ("#E07A2F", "^", "Logistic regression"),
        "hgb": ("#2E8B57", "o", "Gradient boosting"),
        "kbdss": ("#2F6FED", "o", "KB-DSS (hybrid GB)"),
    }
    for key, (color, marker, label) in styles.items():
        mean = np.array(curve[key]["mean"])
        lo = np.array(curve[key]["lo"])
        hi = np.array(curve[key]["hi"])
        ax.plot(sizes, mean, color=color, marker=marker, label=label, lw=1.8)
        ax.fill_between(sizes, lo, hi, color=color, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("Training-set size (firms, log scale)")
    ax.set_ylabel("Macro-F1 on 400 held-out firms")
    ax.set_title("(a) Learning curves")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    phi_levels = heat["phi"]
    grid = np.array(heat["grid"])
    norm = TwoSlopeNorm(vmin=-0.06, vcenter=0.0, vmax=0.06)
    im = ax.imshow(grid, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(phi_levels)))
    ax.set_yticklabels([f"{p:.2f}" for p in phi_levels])
    ax.set_xlabel("Training-set size (firms)")
    ax.set_ylabel("Knowledge fidelity φ")
    ax.set_title("(b) Gain of KB-DSS over plain GB (Δ Macro-F1)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            bold = heat["sig"][i][j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=7, fontweight="bold" if bold else "normal",
                    color="white" if abs(val) > 0.03 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    colors = ["#2F6FED" if k else "#B0B0B0" for k in is_knowledge]
    ypos = np.arange(len(imp_names))[::-1]
    ax.barh(ypos, imp_vals, color=colors[::-1] if False else colors)
    ax.set_yticks(np.arange(len(imp_names)))
    ax.set_yticklabels(imp_names)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (drop in Macro-F1)")
    ax.set_title("(c) What drives KB-DSS")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def explain_one(x_row, log_size, sector, clf) -> dict:
    raw = raw_joint(x_row[None, :], np.array([log_size]), np.array([sector]))
    k = kappa(x_row[None, :])
    feats = np.hstack([raw, k])
    proba = clf.predict_proba(feats)[0]
    pred = int(np.argmax(proba))
    p = k[0, :5]
    mu = k[0, 5:11]
    weak = np.argsort(p)[:2]
    fired = [(RULES[r][0], RULES[r][1], float(mu[r])) for r in range(6) if mu[r] > 0.4]
    actions = []
    for w in weak:
        actions.append(f"Strengthen the {PILLAR_SPEC[w][0]} pillar first.")
    for name, sign, act in fired:
        if sign < 0:
            actions.append(f"Mitigate fired negative rule {name} (activation {act:.2f}).")
    return {
        "class": CLASS_NAMES[pred],
        "probability": float(proba[pred]),
        "weakest_pillars": [PILLAR_SPEC[w][0] for w in weak],
        "rules": fired,
        "actions": actions,
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce the KB-DSS study.")
    parser.add_argument("--data", type=Path, default=None, help="Optional real-firm CSV.")
    parser.add_argument("--misspec-only", action="store_true", help="Run only Table 3 failure-mode sweep.")
    args = parser.parse_args()
    t0 = time.time()
    self_test()
    OUT.mkdir(exist_ok=True)

    if args.data:
        x, log_size, sector, y = load_real(args.data)
        print(f"Loaded {len(y)} real firms from {args.data}")
    else:
        x, log_size, sector, y, _ = generate_benchmark()
        counts = [(y == c).sum() for c in range(3)]
        print(f"Synthetic benchmark: {len(y)} firms, class counts {counts} (target 600/800/600)")

    if args.misspec_only:
        print("\n=== Table 3 only ===")
        misspec_kinds = [
            "correct",
            "flip_three_signs",
            "flip_all_signs",
            "weight_noise",
            "drop_aligned_rules",
            "drop_extra_rules",
        ]
        misspec_sizes = [60, 240, 1500]
        table3 = {k: {} for k in misspec_kinds}
        n_hold = 400
        for n_tr in misspec_sizes:
            by_kind = {k: [] for k in misspec_kinds}
            hgb_scores = []
            for split in range(12):
                rng = np.random.default_rng(SEED + 7000 + split)
                idx = np.arange(len(y))
                tr_pool, te = train_test_split(idx, test_size=n_hold, stratify=y, random_state=SEED + 50 + split)
                sub = rng.choice(tr_pool, size=n_tr, replace=False)
                raw_tr = raw_joint(x[sub], log_size[sub], sector[sub])
                raw_te = raw_joint(x[te], log_size[te], sector[te])
                y_tr, y_te = y[sub], y[te]
                seed = SEED + 11 + split
                pred_h, _ = fit_predict("hgb", raw_tr, y_tr, raw_te, seed)
                hgb_f1 = metrics(y_te, pred_h)["macro_f1"]
                hgb_scores.append(hgb_f1)
                for kind in misspec_kinds:
                    layer = corrupt_layer(kind, np.random.default_rng(SEED + 9000 + split * 10 + n_tr))
                    ft = np.hstack([raw_tr, kappa(x[sub], layer)])
                    fe = np.hstack([raw_te, kappa(x[te], layer)])
                    pred, _ = fit_predict("kbdss", ft, y_tr, fe, seed)
                    by_kind[kind].append(metrics(y_te, pred)["macro_f1"] - hgb_f1)
            print(f"n={n_tr:4d}  GB {np.mean(hgb_scores):.3f}")
            for kind in misspec_kinds:
                g = np.array(by_kind[kind])
                se = g.std(ddof=1) / np.sqrt(len(g))
                table3[kind][str(n_tr)] = {
                    "mean": float(g.mean()),
                    "lo": float(g.mean() - 1.96 * se),
                    "hi": float(g.mean() + 1.96 * se),
                }
                row = table3[kind][str(n_tr)]
                print(f"  {kind:22s}  {row['mean']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]")
        (OUT / "table3.json").write_text(json.dumps(table3, indent=2))
        print(f"Wrote {OUT / 'table3.json'} in {time.time() - t0:.1f}s")
        return

    print("\n=== Table 1: 5x3 stratified CV ===")
    cv = run_cv(x, log_size, sector, y)
    summary = {k: summarize_folds(v) for k, v in cv.items()}
    n_test = len(y) // 5
    n_train = len(y) - n_test
    kbdss_f1 = np.array(summary["kbdss"]["f1_folds"])
    print(f"{'Model':42s}  Accuracy           Macro-F1           QWK                Gain vs row [95% CI]")
    table1 = {}
    order = [
        ("expert", "Expert index (zero training)"),
        ("lr", "Logistic regression"),
        ("rf", "Random forest"),
        ("hgb", "Gradient boosting"),
        ("kblr", "KB-LR (logistic + knowledge)"),
        ("kbdss", "KB-DSS (proposed)"),
    ]
    for key, label in order:
        s = summary[key]
        gain = None
        if key != "kbdss":
            gain = nadeau_bengio(kbdss_f1 - np.array(s["f1_folds"]), n_train, n_test)
        print(fmt_row(label, s, gain))
        table1[key] = {**s, "gain": gain}

    print("\n=== Table 2: ablation and stress tests ===")
    variants = [
        ("raw", "Raw features only (no knowledge)"),
        ("pillars", "+ pillar scores only"),
        ("rules", "+ fuzzy rules only"),
        ("index", "+ expert index only"),
        ("full", "Full KB-DSS (pillars + rules + index)"),
    ]
    raw_mean = summary["hgb"]["macro_f1_mean"]
    table2 = {}
    for comp, label in variants:
        if comp == "full":
            s = summary["kbdss"]
            f1 = kbdss_f1
        elif comp == "raw":
            s = summary["hgb"]
            f1 = np.array(s["f1_folds"])
        else:
            res = run_cv(x, log_size, sector, y, components=comp)
            s = summarize_folds(res["kbdss"])
            f1 = np.array(s["f1_folds"])
        delta = None if comp == "raw" else nadeau_bengio(f1 - np.array(summary["hgb"]["f1_folds"]), n_train, n_test)
        table2[comp] = {**s, "delta": delta}
        extra = "" if delta is None else f"  {delta['mean']:+.3f} [{delta['ci_lo']:+.3f}, {delta['ci_hi']:+.3f}]"
        print(f"{label:42s}  {s['macro_f1_mean']:.3f} ± {s['macro_f1_sd']:.3f}  {s['qwk_mean']:.3f} ± {s['qwk_sd']:.3f}{extra}")

    print("\nStress: fully misspecified knowledge (φ = 0)")
    res0 = run_cv(x, log_size, sector, y, phi=0.0)
    s0 = summarize_folds(res0["kbdss"])
    d0 = nadeau_bengio(np.array(s0["f1_folds"]) - np.array(summary["hgb"]["f1_folds"]), n_train, n_test)
    table2["phi0"] = {**s0, "delta": d0}
    print(fmt_row("KB-DSS φ=0", s0, d0))

    print("Stress: 15% label noise")
    resn = run_cv(x, log_size, sector, y, label_noise=0.15)
    sn_raw = summarize_folds(resn["hgb"])
    sn_kb = summarize_folds(resn["kbdss"])
    dn_raw = nadeau_bengio(np.array(sn_raw["f1_folds"]) - np.array(summary["hgb"]["f1_folds"]), n_train, n_test)
    dn_kb = nadeau_bengio(np.array(sn_kb["f1_folds"]) - np.array(summary["hgb"]["f1_folds"]), n_train, n_test)
    table2["noise_raw"] = {**sn_raw, "delta": dn_raw}
    table2["noise_kbdss"] = {**sn_kb, "delta": dn_kb}
    print(fmt_row("Raw GB + 15% noise", sn_raw, dn_raw))
    print(fmt_row("KB-DSS + 15% noise", sn_kb, dn_kb))

    print("\n=== Learning curves and fidelity map (12 splits) ===")
    sizes = [60, 120, 240, 480, 960, 1500]
    phi_levels = [1.0, 0.75, 0.50, 0.25, 0.0]
    curve = {k: {"mean": [], "lo": [], "hi": []} for k in ["expert", "lr", "hgb", "kbdss"]}
    curve["sizes"] = sizes
    heat_grid = np.zeros((len(phi_levels), len(sizes)))
    heat_lo = np.zeros((len(phi_levels), len(sizes)))
    heat_hi = np.zeros((len(phi_levels), len(sizes)))
    heat_sig = [[False] * len(sizes) for _ in phi_levels]
    n_hold = 400
    for si, n_tr in enumerate(sizes):
        scores = {k: [] for k in curve if k != "sizes"}
        gains = {phi: [] for phi in phi_levels}
        for split in range(12):
            rng = np.random.default_rng(SEED + 1000 + split)
            idx = np.arange(len(y))
            tr_pool, te = train_test_split(idx, test_size=n_hold, stratify=y, random_state=SEED + split)
            sub = rng.choice(tr_pool, size=n_tr, replace=False)
            raw_tr = raw_joint(x[sub], log_size[sub], sector[sub])
            raw_te = raw_joint(x[te], log_size[te], sector[te])
            k_tr, k_te = kappa(x[sub]), kappa(x[te])
            full_tr, full_te = np.hstack([raw_tr, k_tr]), np.hstack([raw_te, k_te])
            y_tr, y_te = y[sub], y[te]
            seed = SEED + split
            for name, tr_x, te_x in [
                ("expert", full_tr, full_te),
                ("lr", raw_tr, raw_te),
                ("hgb", raw_tr, raw_te),
                ("kbdss", full_tr, full_te),
            ]:
                pred, _ = fit_predict(name, tr_x, y_tr, te_x, seed)
                scores[name].append(metrics(y_te, pred)["macro_f1"])
            hgb_f1 = scores["hgb"][-1]
            for pi, phi in enumerate(phi_levels):
                rng_phi = np.random.default_rng(SEED + 5000 + split * 20 + pi)
                xk_tr = apply_fidelity(x[sub], phi, rng_phi)
                xk_te = apply_fidelity(x[te], phi, rng_phi)
                ft = np.hstack([raw_tr, kappa(xk_tr)])
                fe = np.hstack([raw_te, kappa(xk_te)])
                pred, _ = fit_predict("kbdss", ft, y_tr, fe, seed)
                gains[phi].append(metrics(y_te, pred)["macro_f1"] - hgb_f1)
        for key in scores:
            arr = np.array(scores[key])
            curve[key]["mean"].append(float(arr.mean()))
            se = arr.std(ddof=1) / np.sqrt(len(arr))
            curve[key]["lo"].append(float(arr.mean() - 1.96 * se))
            curve[key]["hi"].append(float(arr.mean() + 1.96 * se))
        for pi, phi in enumerate(phi_levels):
            g = np.array(gains[phi])
            se = g.std(ddof=1) / np.sqrt(len(g))
            heat_grid[pi, si] = float(g.mean())
            heat_lo[pi, si] = float(g.mean() - 1.96 * se)
            heat_hi[pi, si] = float(g.mean() + 1.96 * se)
            heat_sig[pi][si] = abs(g.mean()) > 1.96 * se
        print(
            f"n={n_tr:4d}  expert {curve['expert']['mean'][-1]:.3f}  "
            f"LR {curve['lr']['mean'][-1]:.3f}  GB {curve['hgb']['mean'][-1]:.3f}  "
            f"KB-DSS {curve['kbdss']['mean'][-1]:.3f}  "
            f"gain@φ=1 {heat_grid[0, si]:+.3f}"
        )

    print("\n=== Permutation importance (one 400-firm hold-out) ===")
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=400, stratify=y, random_state=SEED)
    raw_tr = raw_joint(x[tr], log_size[tr], sector[tr])
    raw_te = raw_joint(x[te], log_size[te], sector[te])
    full_tr = np.hstack([raw_tr, kappa(x[tr])])
    full_te = np.hstack([raw_te, kappa(x[te])])
    clf = make_hgb(SEED)
    clf.fit(full_tr, y[tr])
    feat_names = INDICATORS + ["log_size", "sector"] + KNOWLEDGE_NAMES
    r = permutation_importance(
        clf, full_te, y[te], n_repeats=20, random_state=SEED, scoring="f1_macro", n_jobs=-1
    )
    order_imp = np.argsort(r.importances_mean)[::-1][:8]
    imp_names = [feat_names[i] for i in order_imp]
    imp_vals = r.importances_mean[order_imp]
    is_knowledge = [i >= 14 for i in order_imp]
    for name, val in zip(imp_names, imp_vals):
        print(f"  {name:28s} {val:.3f}")

    demo = explain_one(x[te][0], log_size[te][0], sector[te][0], clf)
    print("\nExample explanation for one held-out firm:")
    print(json.dumps(demo, indent=2))

    draw_fig1(OUT / "fig1.png")
    draw_fig2(
        OUT / "fig2.png",
        curve,
        {"phi": phi_levels, "grid": heat_grid.tolist(), "sig": heat_sig},
        imp_names,
        imp_vals.tolist(),
        is_knowledge,
    )

    payload = {
        "table1": table1,
        "table2": table2,
        "learning_curve": curve,
        "fidelity": {
            "phi": phi_levels,
            "sizes": sizes,
            "grid": heat_grid.tolist(),
            "lo": heat_lo.tolist(),
            "hi": heat_hi.tolist(),
            "sig": heat_sig,
        },
        "permutation_importance": {
            "names": imp_names,
            "values": [float(v) for v in imp_vals],
        },
        "example": demo,
        "seconds": time.time() - t0,
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    (OUT / "metrics.json").write_text(json.dumps(_clean(payload), indent=2))
    print(f"\nWrote {OUT / 'fig1.png'}, {OUT / 'fig2.png'}, {OUT / 'metrics.json'}")
    print(f"Elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
