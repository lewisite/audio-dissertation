"""
Experiment Analysis — Main Orchestrator
=========================================
Dissertation: "The Effect of Explainability on End-User Trust in Neural Audio Codecs"

Loads participant data from a CSV, runs all statistical tests from Chapter 3,
generates dissertation-ready figures, and writes a full analysis report.

CSV format (one row per participant × condition):
  participant_id  : string (e.g. P001)
  condition_order : string (AB = saw Codec A first, BA = saw Codec B first)
  codec           : string (A or B)
  latency_ms      : int (50 or 150)
  trust_q1..q12   : int (1-7 Likert responses)
  mos_rating      : float (1.0-5.0 Mean Opinion Score)
  audio_clip_id   : string (which audio file was played)
  session_date    : string (YYYY-MM-DD)

Usage:
  # Run on synthetic data (for testing and methodology validation):
  python src/analyze_experiment.py --synthetic

  # Run on real participant data:
  python src/analyze_experiment.py --input data/experiment_responses.csv

  # Specify output directory:
  python src/analyze_experiment.py --input data/experiment_responses.csv --output-dir analysis/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from stats import (
    score_trust_scale, cronbach_alpha, descriptive_stats,
    shapiro_wilk, cohen_d, paired_ttest, wilcoxon_signed_rank,
    pearson_correlation, repeated_measures_anova, evaluate_hypothesis,
    TRUST_ITEMS, TRUST_SCALE_MIN, TRUST_SCALE_MAX,
)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_participants: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic participant data for methodology testing.

    Design mirrors Chapter 3: within-subjects, 2x2 (codec x latency),
    counterbalanced condition order.

    Expected effects built into the data:
      - Trust: Codec A significantly higher than Codec B (H1 supported)
      - MOS: no significant difference between Codec A and B (H2 supported)
      - Latency: moderate negative effect on trust (higher latency = lower trust)

    This lets you validate the entire analysis pipeline before real data arrives.
    Replace this dataset with real CSV data when participants are collected.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # True effect parameters (used to generate realistic synthetic data)
    # Trust: Codec A mean = 62, Codec B mean = 48 (out of 84)
    # Low latency adds +3 points, moderate latency subtracts -3 points
    trust_params = {
        ("A", 50):  {"mu": 65, "sigma": 8},
        ("A", 150): {"mu": 59, "sigma": 9},
        ("B", 50):  {"mu": 51, "sigma": 9},
        ("B", 150): {"mu": 45, "sigma": 10},
    }
    mos_params = {
        ("A", 50):  {"mu": 4.10, "sigma": 0.55},
        ("A", 150): {"mu": 3.95, "sigma": 0.60},
        ("B", 50):  {"mu": 3.90, "sigma": 0.60},
        ("B", 150): {"mu": 3.75, "sigma": 0.65},
    }

    # Participant-level random effects (some people generally trust more or less)
    participant_baseline = rng.normal(0, 5, n_participants)

    # Counterbalance condition order: half see A first, half see B first
    condition_orders = (["AB"] * (n_participants // 2) +
                        ["BA"] * (n_participants - n_participants // 2))
    rng.shuffle(condition_orders)

    audio_clips = ["speech_01", "speech_02", "music_01", "noise_01", "tonal_01"]

    for i in range(n_participants):
        pid       = f"P{i+1:03d}"
        p_bias    = participant_baseline[i]
        order     = condition_orders[i]

        for codec in ["A", "B"]:
            for latency in [50, 150]:
                params  = trust_params[(codec, latency)]
                mos_p   = mos_params[(codec, latency)]

                # Generate correlated trust score (total, then back-derive items)
                raw_total = rng.normal(params["mu"] + p_bias, params["sigma"])
                raw_total = float(np.clip(raw_total, TRUST_SCALE_MIN, TRUST_SCALE_MAX))

                # Simulate 12 Likert responses consistent with the total score
                # (distribute total across 12 items with item-level noise)
                mean_per_item = raw_total / 12.0
                # Items are on 1-7 scale; un-reverse positive items
                trust_items_raw = rng.normal(mean_per_item, 0.5, 12)
                trust_items_raw = np.clip(np.round(trust_items_raw), 1, 7).astype(int)

                # Reverse-score items 1-5 back to raw (so scoring gives the right total)
                # stored_q = raw for positive items; for reverse items, stored = 8 - reversed
                stored = trust_items_raw.copy().astype(float)
                # Reverse items 0-4: stored as (8 - item_value) so scoring gives item_value back
                stored[0:5] = 8.0 - stored[0:5]
                stored = np.clip(stored, 1, 7).astype(int)

                # MOS rating (1.0-5.0)
                mos = float(np.clip(rng.normal(mos_p["mu"], mos_p["sigma"]), 1.0, 5.0))
                mos = round(mos * 2) / 2    # round to nearest 0.5 (typical MOS scale)

                row = {
                    "participant_id": pid,
                    "condition_order": order,
                    "codec":          codec,
                    "latency_ms":     latency,
                    "mos_rating":     mos,
                    "audio_clip_id":  rng.choice(audio_clips),
                    "session_date":   "2026-03-01",   # placeholder
                }
                for q in range(12):
                    row[f"trust_q{q+1}"] = int(stored[q])

                rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING AND VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = (
    ["participant_id", "condition_order", "codec", "latency_ms", "mos_rating"] +
    [f"trust_q{i}" for i in range(1, 13)]
)

TRUST_Q_COLS = [f"trust_q{i}" for i in range(1, 13)]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate experiment CSV."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def validate_data(df: pd.DataFrame) -> list:
    """
    Check for data quality issues. Returns list of warning strings.
    Does NOT raise errors — reports issues for researcher review.
    """
    warnings = []

    # Check Likert range
    for q in TRUST_Q_COLS:
        out = df[(df[q] < 1) | (df[q] > 7)]
        if len(out):
            warnings.append(f"Column {q}: {len(out)} out-of-range values (should be 1-7)")

    # Check MOS range
    mos_out = df[(df["mos_rating"] < 1.0) | (df["mos_rating"] > 5.0)]
    if len(mos_out):
        warnings.append(f"mos_rating: {len(mos_out)} out-of-range values (should be 1.0-5.0)")

    # Check codec values
    bad_codec = df[~df["codec"].isin(["A", "B"])]
    if len(bad_codec):
        warnings.append(f"codec column: unexpected values {df['codec'].unique()}")

    # Check latency values
    bad_lat = df[~df["latency_ms"].isin([50, 150])]
    if len(bad_lat):
        warnings.append(f"latency_ms: unexpected values {df['latency_ms'].unique()}")

    # Check for missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        warnings.append(f"Missing values detected: {missing.to_dict()}")

    # Check each participant has all 4 conditions
    n_conds = df.groupby("participant_id").size()
    wrong = n_conds[n_conds != 4]
    if len(wrong):
        warnings.append(f"Participants without all 4 conditions: {list(wrong.index)}")

    return warnings


def _add_trust_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite trust scores and add as a column."""
    trust_matrix = df[TRUST_Q_COLS].values.astype(float)
    df = df.copy()
    df["trust_score"] = score_trust_scale(trust_matrix)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(df: pd.DataFrame) -> dict:
    """
    Run all statistical tests from Chapter 3. Returns structured results dict.
    """
    df = _add_trust_scores(df)
    results = {}

    # ── 1. RELIABILITY — Cronbach's alpha for the trust scale ────────────────
    # Use all trust responses (both conditions) to assess scale reliability
    trust_matrix = df[TRUST_Q_COLS].values.astype(float)
    # Reverse-score items 1-5 before computing alpha
    trust_matrix_scored = trust_matrix.copy()
    trust_matrix_scored[:, :5] = 8.0 - trust_matrix_scored[:, :5]
    results["reliability"] = cronbach_alpha(trust_matrix_scored)

    # ── 2. DESCRIPTIVE STATISTICS ────────────────────────────────────────────
    codec_a   = df[df["codec"] == "A"]
    codec_b   = df[df["codec"] == "B"]
    low_lat   = df[df["latency_ms"] == 50]
    mod_lat   = df[df["latency_ms"] == 150]

    results["descriptives"] = {
        "trust_codec_a":     descriptive_stats(codec_a["trust_score"].values, "Trust — Codec A"),
        "trust_codec_b":     descriptive_stats(codec_b["trust_score"].values, "Trust — Codec B"),
        "trust_low_latency": descriptive_stats(low_lat["trust_score"].values, "Trust — Low latency (50ms)"),
        "trust_mod_latency": descriptive_stats(mod_lat["trust_score"].values, "Trust — Moderate latency (150ms)"),
        "mos_codec_a":       descriptive_stats(codec_a["mos_rating"].values, "MOS — Codec A"),
        "mos_codec_b":       descriptive_stats(codec_b["mos_rating"].values, "MOS — Codec B"),
    }

    # ── 3. PAIRED DATA — one score per participant per condition ─────────────
    # Pivot to wide format: one row per participant
    wide_trust = df.pivot_table(
        index="participant_id", columns="codec", values="trust_score", aggfunc="mean"
    ).reset_index()
    wide_mos = df.pivot_table(
        index="participant_id", columns="codec", values="mos_rating", aggfunc="mean"
    ).reset_index()

    trust_a_paired = wide_trust["A"].values
    trust_b_paired = wide_trust["B"].values
    mos_a_paired   = wide_mos["A"].values
    mos_b_paired   = wide_mos["B"].values

    # ── 4. NORMALITY TESTS ───────────────────────────────────────────────────
    trust_diff = trust_a_paired - trust_b_paired
    mos_diff   = mos_a_paired   - mos_b_paired

    results["normality"] = {
        "trust_codec_a":  shapiro_wilk(trust_a_paired,   "Trust scores — Codec A"),
        "trust_codec_b":  shapiro_wilk(trust_b_paired,   "Trust scores — Codec B"),
        "trust_diff":     shapiro_wilk(trust_diff,        "Trust difference scores (A - B)"),
        "mos_codec_a":    shapiro_wilk(mos_a_paired,      "MOS ratings — Codec A"),
        "mos_codec_b":    shapiro_wilk(mos_b_paired,      "MOS ratings — Codec B"),
        "mos_diff":       shapiro_wilk(mos_diff,           "MOS difference scores (A - B)"),
    }

    # ── 5. RQ1 — Trust: Codec A vs Codec B ───────────────────────────────────
    results["rq1_trust_ttest"] = paired_ttest(
        trust_a_paired, trust_b_paired,
        "Codec A (explainable)", "Codec B (black box)"
    )
    results["rq1_trust_wilcoxon"] = wilcoxon_signed_rank(
        trust_a_paired, trust_b_paired,
        "Codec A (explainable)", "Codec B (black box)"
    )
    results["rq1_hypothesis"] = evaluate_hypothesis(
        results["rq1_trust_ttest"],
        results["rq1_trust_wilcoxon"],
        hypothesis="H1",
        null="End-user trust does not differ between Codec A and Codec B.",
        alternative="End-user trust is significantly higher with Codec A (explainable) than Codec B (black box).",
    )

    # ── 6. RQ2 — Audio quality (MOS): Codec A vs Codec B ────────────────────
    results["rq2_mos_ttest"] = paired_ttest(
        mos_a_paired, mos_b_paired,
        "Codec A (explainable)", "Codec B (black box)"
    )
    results["rq2_mos_wilcoxon"] = wilcoxon_signed_rank(
        mos_a_paired, mos_b_paired,
        "Codec A (explainable)", "Codec B (black box)"
    )
    results["rq2_hypothesis"] = evaluate_hypothesis(
        results["rq2_mos_ttest"],
        results["rq2_mos_wilcoxon"],
        hypothesis="H2",
        null="Codec A and Codec B show no significant difference in audio quality (MOS).",
        alternative="There is a significant difference in MOS between Codec A and Codec B.",
    )

    # ── 7. CORRELATION — Trust vs MOS ────────────────────────────────────────
    all_trust = df["trust_score"].values
    all_mos   = df["mos_rating"].values
    results["trust_mos_correlation"] = pearson_correlation(
        all_trust, all_mos, "Trust score", "MOS rating"
    )

    # ── 8. ANOVA — Codec × Latency interaction ───────────────────────────────
    anova_data = {
        "participant": df["participant_id"].tolist(),
        "codec":       df["codec"].tolist(),
        "latency":     df["latency_ms"].astype(str).tolist(),
        "trust_score": df["trust_score"].tolist(),
    }
    results["anova_codec_latency"] = repeated_measures_anova(anova_data)

    # ── 9. ORDER EFFECT CHECK — was there a practice/fatigue effect? ─────────
    first_codec_a  = df[(df["condition_order"] == "AB") & (df["codec"] == "A")]["trust_score"].values
    second_codec_a = df[(df["condition_order"] == "BA") & (df["codec"] == "A")]["trust_score"].values
    if len(first_codec_a) > 1 and len(second_codec_a) > 1:
        order_test = scipy_stats.ttest_ind(first_codec_a, second_codec_a)
        results["order_effect_check"] = {
            "test":        "Independent t-test on Codec A trust by condition order",
            "t_statistic": float(order_test.statistic),
            "p_value":     float(order_test.pvalue),
            "significant": bool(order_test.pvalue < 0.05),
            "note":        ("Non-significant order effect supports internal validity of counterbalancing."
                            if order_test.pvalue >= 0.05 else
                            "Significant order effect detected -- investigate carryover or learning."),
        }

    return results, df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def generate_figures(df: pd.DataFrame, results: dict, out_dir: Path):
    """Generate all dissertation figures from experiment data."""

    palette = {"A": "steelblue", "B": "darkorange"}
    lat_palette = {50: "seagreen", 150: "crimson"}

    # ── Figure 1: Trust scores — Codec A vs B (boxplot + individual points) ──
    fig, ax = plt.subplots(figsize=(8, 6))
    codec_groups = [
        df[df["codec"] == "A"]["trust_score"].values,
        df[df["codec"] == "B"]["trust_score"].values,
    ]
    bp = ax.boxplot(codec_groups, patch_artist=True, widths=0.45,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], ["steelblue", "darkorange"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (group, color) in enumerate(zip(codec_groups, ["steelblue", "darkorange"])):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(group))
        ax.scatter(np.full(len(group), i + 1) + jitter, group,
                   color=color, alpha=0.5, s=30, zorder=3)

    t_res = results["rq1_trust_ttest"]
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"Codec A\n(Explainable)\nM={t_res['mean_a']:.1f}",
        f"Codec B\n(Black Box)\nM={t_res['mean_b']:.1f}",
    ], fontsize=10)
    ax.set_ylabel("Trust Score (Jian et al., 2000)\nRange: 12-84", fontsize=10)
    ax.set_ylim(TRUST_SCALE_MIN - 5, TRUST_SCALE_MAX + 5)
    ax.set_title(
        f"Trust Scores: Codec A vs Codec B\n{t_res['apa_string']}",
        fontsize=11, fontweight="bold"
    )
    ax.axhline(TRUST_SCALE_MAX / 2, color="grey", linestyle=":", alpha=0.5,
               label="Scale midpoint (48)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Significance bracket
    if t_res["significant"]:
        y_max = max(max(codec_groups[0]), max(codec_groups[1])) + 3
        ax.plot([1, 1, 2, 2], [y_max, y_max + 1, y_max + 1, y_max],
                color="black", linewidth=1.5)
        p = t_res["p_value"]
        sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ax.text(1.5, y_max + 1.5, sig_str, ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_trust_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: MOS — Codec A vs B ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    mos_groups = [
        df[df["codec"] == "A"]["mos_rating"].values,
        df[df["codec"] == "B"]["mos_rating"].values,
    ]
    bp = ax.boxplot(mos_groups, patch_artist=True, widths=0.45,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], ["steelblue", "darkorange"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (group, color) in enumerate(zip(mos_groups, ["steelblue", "darkorange"])):
        jitter = np.random.default_rng(99).uniform(-0.12, 0.12, len(group))
        ax.scatter(np.full(len(group), i + 1) + jitter, group,
                   color=color, alpha=0.5, s=30, zorder=3)

    m_res = results["rq2_mos_ttest"]
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"Codec A\n(Explainable)\nM={m_res['mean_a']:.2f}",
        f"Codec B\n(Black Box)\nM={m_res['mean_b']:.2f}",
    ], fontsize=10)
    ax.set_ylabel("Mean Opinion Score (MOS)\nRange: 1.0-5.0", fontsize=10)
    ax.set_ylim(0.5, 5.5)
    ax.set_title(
        f"Perceived Audio Quality (MOS): Codec A vs Codec B\n{m_res['apa_string']}",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_mos_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 3: Trust by codec × latency (2×2 interaction plot) ────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for codec, color, marker in [("A", "steelblue", "o"), ("B", "darkorange", "s")]:
        means, cis = [], []
        for lat in [50, 150]:
            vals = df[(df["codec"] == codec) & (df["latency_ms"] == lat)]["trust_score"].values
            means.append(np.mean(vals))
            cis.append(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar([50, 150], means, yerr=cis, marker=marker, color=color,
                    linewidth=2, markersize=8, capsize=5,
                    label=f"Codec {'A (Explainable)' if codec=='A' else 'B (Black Box)'}")

    ax.set_xlabel("Latency Condition (ms)", fontsize=10)
    ax.set_ylabel("Mean Trust Score (±95% CI)", fontsize=10)
    ax.set_xticks([50, 150])
    ax.set_xticklabels(["Low (50 ms)", "Moderate (150 ms)"])
    ax.set_title("Trust by Codec and Latency Condition\n(Codec × Latency Interaction)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(TRUST_SCALE_MIN - 5, TRUST_SCALE_MAX + 5)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_trust_interaction.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 4: Trust vs MOS scatter (correlation) ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    for codec, color in [("A", "steelblue"), ("B", "darkorange")]:
        sub = df[df["codec"] == codec]
        ax.scatter(sub["trust_score"].values, sub["mos_rating"].values,
                   color=color, alpha=0.5, s=40,
                   label=f"Codec {'A (Explainable)' if codec=='A' else 'B (Black Box)'}")

    # Regression line
    x_all = df["trust_score"].values
    y_all = df["mos_rating"].values
    m, b  = np.polyfit(x_all, y_all, 1)
    x_line = np.linspace(x_all.min(), x_all.max(), 100)
    ax.plot(x_line, m * x_line + b, "k--", linewidth=1.5, alpha=0.7, label="Regression line")

    r_res = results["trust_mos_correlation"]
    ax.set_xlabel("Trust Score (Jian et al., 2000)", fontsize=10)
    ax.set_ylabel("Mean Opinion Score (MOS)", fontsize=10)
    ax.set_title(
        f"Trust Score vs Perceived Audio Quality (MOS)\n{r_res['apa_string']}",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_trust_mos_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 5: Q-Q plots for normality check ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Normal Q-Q Plots — Trust Difference Scores and MOS Differences",
                 fontsize=11, fontweight="bold")

    wide_trust = df.pivot_table(
        index="participant_id", columns="codec", values="trust_score", aggfunc="mean"
    ).reset_index()
    trust_diff = (wide_trust["A"] - wide_trust["B"]).values

    wide_mos = df.pivot_table(
        index="participant_id", columns="codec", values="mos_rating", aggfunc="mean"
    ).reset_index()
    mos_diff = (wide_mos["A"] - wide_mos["B"]).values

    for ax, diff, title in [
        (axes[0], trust_diff, "Trust Difference (Codec A - Codec B)"),
        (axes[1], mos_diff,   "MOS Difference (Codec A - Codec B)"),
    ]:
        (osm, osr), (slope, intercept, r) = scipy_stats.probplot(diff, dist="norm")
        ax.scatter(osm, osr, color="steelblue", s=30, alpha=0.7)
        ax.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5, label="Normal line")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_qq_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"      Saved 5 figures to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# DISSERTATION-READY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results: dict, df: pd.DataFrame, out_dir: Path, is_synthetic: bool = False):
    """
    Write a full analysis report formatted for direct use in the dissertation.
    Text is structured to match Chapter 4 (Results) conventions.
    """
    n = df["participant_id"].nunique()
    lines = [
        "=" * 70,
        "  DISSERTATION ANALYSIS REPORT",
        "  Chapter 4: Results",
        "=" * 70,
    ]
    if is_synthetic:
        lines += [
            "",
            "  NOTE: This report is based on SYNTHETIC data generated for",
            "  methodology testing. Replace with real participant data before",
            "  including in the dissertation.",
        ]
    lines += [
        "",
        f"  N = {n} participants  |  Design: 2x2 within-subjects",
        f"  Factors: Codec (A/B) x Latency (50ms/150ms)",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    sep = "-" * 70

    # ── Reliability ───────────────────────────────────────────────────────────
    rel = results["reliability"]
    lines += [
        sep,
        "SECTION 1: SCALE RELIABILITY",
        sep,
        f"Cronbach's alpha was computed for the 12-item Jian et al. (2000)",
        f"Trust in Automation Scale across all participant responses.",
        f"",
        f"  alpha = {rel['alpha']:.3f}  ({rel['interpretation']})",
        f"  N items = {rel['n_items']}",
        f"  Meets dissertation threshold (>= 0.80): {'YES' if rel['meets_threshold'] else 'NO -- interpret with caution'}",
        f"",
        f"Alpha-if-item-deleted:",
    ]
    for i, a_del in enumerate(rel["if_item_deleted"]):
        flag = "  <-- removing this item would increase alpha" if a_del > rel["alpha"] else ""
        lines.append(f"  Q{i+1:>2} ({TRUST_ITEMS[i][:45]:<45}): {a_del:.3f}{flag}")
    lines.append("")

    # ── Descriptives ──────────────────────────────────────────────────────────
    lines += [sep, "SECTION 2: DESCRIPTIVE STATISTICS", sep, ""]

    desc_keys = [
        ("trust_codec_a",     "Trust Score — Codec A (Explainable)"),
        ("trust_codec_b",     "Trust Score — Codec B (Black Box)"),
        ("trust_low_latency", "Trust Score — Low Latency (50 ms)"),
        ("trust_mod_latency", "Trust Score — Moderate Latency (150 ms)"),
        ("mos_codec_a",       "MOS — Codec A (Explainable)"),
        ("mos_codec_b",       "MOS — Codec B (Black Box)"),
    ]
    lines.append(f"  {'Measure':<45} {'N':>4} {'M':>7} {'SD':>7} {'Mdn':>7} {'IQR':>7} {'95% CI':>18}")
    lines.append(f"  {'-'*45} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*18}")
    for key, label in desc_keys:
        d = results["descriptives"][key]
        ci = f"[{d['ci95_low']:.2f}, {d['ci95_high']:.2f}]"
        lines.append(
            f"  {label:<45} {d['n']:>4} {d['mean']:>7.2f} {d['sd']:>7.2f} "
            f"{d['median']:>7.2f} {d['iqr']:>7.2f} {ci:>18}"
        )
    lines.append("")

    # ── Normality ─────────────────────────────────────────────────────────────
    lines += [sep, "SECTION 3: NORMALITY TESTS (Shapiro-Wilk)", sep, ""]
    for key, label in [
        ("trust_diff", "Trust difference scores (A - B)"),
        ("mos_diff",   "MOS difference scores (A - B)"),
    ]:
        n_res = results["normality"][key]
        lines.append(f"  {label}: {n_res['interpretation']}")
    lines.append("")

    # ── RQ1 ───────────────────────────────────────────────────────────────────
    lines += [sep, "SECTION 4: RQ1 -- EFFECT OF EXPLAINABILITY ON TRUST", sep, ""]
    lines.append("H1 (null): Trust does not differ between Codec A and Codec B.")
    lines.append("H1 (alt):  Trust is significantly higher with Codec A (explainable).")
    lines.append("")

    t1 = results["rq1_trust_ttest"]
    w1 = results["rq1_trust_wilcoxon"]
    h1 = results["rq1_hypothesis"]

    lines += [
        "  Paired t-test:",
        f"    {t1['conclusion']}",
        "",
        "  Wilcoxon signed-rank (robustness check):",
        f"    {w1['conclusion']}",
        "",
        f"  HYPOTHESIS DECISION: {h1['decision']}",
        f"  {h1['summary']}",
        "",
        f"  APA format: {t1['apa_string']}",
    ]
    lines.append("")

    # ── RQ2 ───────────────────────────────────────────────────────────────────
    lines += [sep, "SECTION 5: RQ2 -- EFFECT ON AUDIO QUALITY (MOS)", sep, ""]
    lines.append("H2 (null): MOS does not differ between Codec A and Codec B.")
    lines.append("H2 (alt):  There is a significant MOS difference between Codec A and Codec B.")
    lines.append("")

    t2 = results["rq2_mos_ttest"]
    w2 = results["rq2_mos_wilcoxon"]
    h2 = results["rq2_hypothesis"]

    lines += [
        "  Paired t-test:",
        f"    {t2['conclusion']}",
        "",
        "  Wilcoxon signed-rank (robustness check):",
        f"    {w2['conclusion']}",
        "",
        f"  HYPOTHESIS DECISION: {h2['decision']}",
        f"  {h2['summary']}",
        "",
        f"  APA format: {t2['apa_string']}",
    ]
    lines.append("")

    # ── Correlation ───────────────────────────────────────────────────────────
    lines += [sep, "SECTION 6: TRUST-QUALITY CORRELATION", sep, ""]
    r_res = results["trust_mos_correlation"]
    lines += [
        "  Pearson correlation between trust scores and MOS ratings:",
        f"  {r_res['conclusion']}",
        f"  R-squared = {r_res['r_squared']:.3f} ({r_res['r_squared']*100:.1f}% shared variance)",
        "",
    ]

    # ── ANOVA ─────────────────────────────────────────────────────────────────
    lines += [sep, "SECTION 7: CODEC x LATENCY INTERACTION (Repeated Measures ANOVA)", sep, ""]
    aov = results["anova_codec_latency"]
    if aov.get("results"):
        for source, res in aov["results"].items():
            sig = "*" if res["significant"] else "ns"
            lines.append(f"  {source:<20}: {res['apa_string']}  [{sig}]")
    else:
        lines.append(f"  {aov.get('note', 'ANOVA results unavailable')}")
    lines.append("")

    # ── Order effect ──────────────────────────────────────────────────────────
    if "order_effect_check" in results:
        oe = results["order_effect_check"]
        lines += [
            sep,
            "SECTION 8: ORDER EFFECT CHECK (internal validity)",
            sep,
            f"  {oe['note']}",
            f"  t = {oe['t_statistic']:.3f}, p = {oe['p_value']:.4f}",
            "",
        ]

    lines += ["=" * 70, "  END OF REPORT", "=" * 70]

    report = "\n".join(lines)
    path   = out_dir / "analysis_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n  Report saved to: {path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main(input_path: Optional[str], use_synthetic: bool, output_dir: str, n_synthetic: int):

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_synthetic" if use_synthetic else "_real")
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  EXPERIMENT ANALYSIS")
    print(sep)

    # ── Load data ─────────────────────────────────────────────────────────────
    if use_synthetic:
        print(f"  Using SYNTHETIC data (N={n_synthetic}) for methodology validation.")
        print(f"  Replace with real participant data when available.\n")
        df = generate_synthetic_data(n_participants=n_synthetic)
        df.to_csv(out_dir / "synthetic_data.csv", index=False)
        print(f"  Synthetic data saved to: {out_dir / 'synthetic_data.csv'}")
        is_synthetic = True
    else:
        print(f"  Loading real participant data from: {input_path}")
        df = load_data(input_path)
        is_synthetic = False

    # ── Validate ──────────────────────────────────────────────────────────────
    warnings = validate_data(df)
    if warnings:
        print(f"\n  DATA QUALITY WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
        print()
    else:
        print(f"  Data validation passed. N={df['participant_id'].nunique()} participants.\n")

    # ── Run analysis ──────────────────────────────────────────────────────────
    print("  Running statistical analysis...\n")
    results, df = run_full_analysis(df)

    # ── Save results JSON ─────────────────────────────────────────────────────
    def _serialise(obj):
        if isinstance(obj, (np.integer, np.int64)):  return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray):               return obj.tolist()
        if isinstance(obj, np.bool_):                 return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_serialise)
    print(f"  Results saved to: {out_dir / 'results.json'}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n  Generating figures...")
    generate_figures(df, results, out_dir)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    write_report(results, df, out_dir, is_synthetic=is_synthetic)

    print(f"\n  All outputs saved to: {out_dir}")
    print(sep + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dissertation experiment analysis -- trust and audio quality"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--synthetic", action="store_true",
        help="Run on synthetic data (for methodology testing)"
    )
    group.add_argument(
        "--input", type=str,
        help="Path to real participant data CSV"
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=30,
        help="Number of synthetic participants (default: 30)"
    )
    parser.add_argument(
        "--output-dir", default="analysis",
        help="Parent directory for outputs (default: analysis/)"
    )
    args = parser.parse_args()
    main(args.input, args.synthetic, args.output_dir, args.n_synthetic)
