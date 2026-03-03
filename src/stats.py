"""
Statistical Analysis Functions
================================
Dissertation: "The Effect of Explainability on End-User Trust in Neural Audio Codecs"

Implements every statistical test specified in Chapter 3 methodology.
All functions return structured dicts so results can be:
  - printed to the analysis report
  - saved to results.json
  - fed into the dashboard

Chapter 3 tests mapped to functions:
  Reliability         -> cronbach_alpha()
  Normality check     -> shapiro_wilk()
  Primary RQ1 test    -> paired_ttest()  (trust Codec A vs B)
  Non-parametric back -> wilcoxon_signed_rank()  (if normality violated)
  Effect size         -> cohen_d()
  ANOVA (latency)     -> repeated_measures_anova()
  ANOVA effect size   -> eta_squared()
  Correlation         -> pearson_correlation()  (trust vs MOS)
  Trust scoring       -> score_trust_scale()
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# TRUST SCALE SCORING
# ─────────────────────────────────────────────────────────────────────────────

# Jian et al. (2000) Trust in Automation Scale — 12 items, Likert 1-7
# Items 1-5 are negatively worded and must be reverse-scored (8 - raw score)
# Items 6-12 are positively worded
# Composite score range: 12 (no trust) to 84 (maximum trust)

TRUST_ITEMS = [
    "The system is deceptive",                                    # Q1  (reverse)
    "The system behaves in an underhanded manner",                # Q2  (reverse)
    "I am suspicious of the system's intent, action, or outputs",# Q3  (reverse)
    "I am wary of the system",                                    # Q4  (reverse)
    "The system's actions will have a harmful or injurious outcome", # Q5 (reverse)
    "I am confident in the system",                               # Q6
    "The system provides security",                               # Q7
    "The system has integrity",                                   # Q8
    "The system is dependable",                                   # Q9
    "The system is reliable",                                     # Q10
    "I can trust the system",                                     # Q11
    "I am familiar with the system",                              # Q12
]
REVERSE_ITEMS = [0, 1, 2, 3, 4]   # 0-indexed
POSITIVE_ITEMS = [5, 6, 7, 8, 9, 10, 11]

TRUST_SCALE_MIN = 12
TRUST_SCALE_MAX = 84


def score_trust_scale(responses: np.ndarray) -> np.ndarray:
    """
    Score the Jian et al. (2000) Trust in Automation Scale.

    Args:
        responses: shape [n_participants, 12], raw Likert responses (1-7)

    Returns:
        composite_scores: shape [n_participants], range 12-84
    """
    data = responses.copy().astype(float)
    data[:, REVERSE_ITEMS] = 8.0 - data[:, REVERSE_ITEMS]
    return data.sum(axis=1)


def score_single_participant(responses_12: list) -> float:
    """Score a single participant's 12-item trust responses."""
    return score_trust_scale(np.array([responses_12]))[0]


# ─────────────────────────────────────────────────────────────────────────────
# RELIABILITY
# ─────────────────────────────────────────────────────────────────────────────

def cronbach_alpha(data: np.ndarray) -> dict:
    """
    Compute Cronbach's alpha internal consistency reliability.

    Chapter 3 target: α ≥ 0.80 ("Good").

    Args:
        data: shape [n_participants, n_items], already reverse-scored if needed

    Returns dict with:
        alpha           : float, Cronbach's alpha coefficient
        n_items         : int
        interpretation  : plain-language quality description
        meets_threshold : bool (threshold = 0.80 per Chapter 3)
        if_item_deleted : list of alpha-if-item-deleted values
    """
    n, k = data.shape
    if k < 2:
        raise ValueError("Need at least 2 items for Cronbach's alpha")

    item_vars  = data.var(axis=0, ddof=1)
    total_var  = data.sum(axis=1).var(ddof=1)
    alpha      = (k / (k - 1)) * (1.0 - item_vars.sum() / total_var)

    # Alpha-if-item-deleted
    alpha_if_deleted = []
    for i in range(k):
        subset   = np.delete(data, i, axis=1)
        sv       = subset.var(axis=0, ddof=1)
        tv       = subset.sum(axis=1).var(ddof=1)
        k2       = k - 1
        a_del    = (k2 / (k2 - 1)) * (1.0 - sv.sum() / tv)
        alpha_if_deleted.append(float(a_del))

    if alpha >= 0.90:
        interp = "Excellent (alpha >= 0.90)"
    elif alpha >= 0.80:
        interp = "Good (0.80 <= alpha < 0.90) -- meets dissertation threshold"
    elif alpha >= 0.70:
        interp = "Acceptable (0.70 <= alpha < 0.80) -- below target but usable"
    elif alpha >= 0.60:
        interp = "Questionable (0.60 <= alpha < 0.70) -- consider removing weak items"
    else:
        interp = "Poor (alpha < 0.60) -- scale reliability is insufficient"

    return {
        "alpha":            float(alpha),
        "n_items":          k,
        "n_participants":   n,
        "interpretation":   interp,
        "meets_threshold":  bool(alpha >= 0.80),
        "if_item_deleted":  alpha_if_deleted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_stats(data: np.ndarray, label: str = "") -> dict:
    """
    Full descriptive statistics for a 1-D array.
    Returns everything needed for a dissertation results table.
    """
    n = len(data)
    q1, q3 = np.percentile(data, [25, 75])
    return {
        "label":    label,
        "n":        n,
        "mean":     float(np.mean(data)),
        "sd":       float(np.std(data, ddof=1)),
        "median":   float(np.median(data)),
        "iqr":      float(q3 - q1),
        "min":      float(np.min(data)),
        "max":      float(np.max(data)),
        "q1":       float(q1),
        "q3":       float(q3),
        "se":       float(np.std(data, ddof=1) / np.sqrt(n)),
        "ci95_low": float(np.mean(data) - 1.96 * np.std(data, ddof=1) / np.sqrt(n)),
        "ci95_high":float(np.mean(data) + 1.96 * np.std(data, ddof=1) / np.sqrt(n)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALITY
# ─────────────────────────────────────────────────────────────────────────────

def shapiro_wilk(data: np.ndarray, label: str = "", alpha: float = 0.05) -> dict:
    """
    Shapiro-Wilk normality test.

    Chapter 3: test normality before choosing parametric vs non-parametric test.
    If p > 0.05, normality assumption is not rejected — use paired t-test.
    If p <= 0.05, normality is violated — use Wilcoxon signed-rank as backup.

    Returns dict with:
        statistic   : W statistic
        p_value     : p-value
        normal      : bool (True if normality NOT rejected at alpha level)
        interpretation : plain-language conclusion
    """
    stat, p = scipy_stats.shapiro(data)
    normal = bool(p > alpha)
    interp = (
        f"W = {stat:.4f}, p = {p:.4f}. "
        + ("Normality assumption NOT rejected (p > 0.05). "
           "Proceed with parametric test (paired t-test)."
           if normal else
           "Normality assumption VIOLATED (p <= 0.05). "
           "Use non-parametric Wilcoxon signed-rank test as primary.")
    )
    return {
        "label":          label,
        "statistic":      float(stat),
        "p_value":        float(p),
        "normal":         normal,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EFFECT SIZE
# ─────────────────────────────────────────────────────────────────────────────

def cohen_d(group_a: np.ndarray, group_b: np.ndarray,
            paired: bool = True) -> dict:
    """
    Cohen's d effect size.

    For paired data (within-subjects), uses the difference scores.
    Interpretation thresholds (Cohen, 1988):
      Small  : d = 0.20
      Medium : d = 0.50
      Large  : d = 0.80

    Chapter 3 reports Cohen's d alongside all t-tests.
    """
    if paired:
        diff = group_a - group_b
        d    = float(np.mean(diff) / np.std(diff, ddof=1))
    else:
        pooled_sd = np.sqrt(
            ((len(group_a) - 1) * np.var(group_a, ddof=1) +
             (len(group_b) - 1) * np.var(group_b, ddof=1)) /
            (len(group_a) + len(group_b) - 2)
        )
        d = float((np.mean(group_a) - np.mean(group_b)) / pooled_sd)

    abs_d = abs(d)
    if abs_d >= 0.80:
        magnitude = "Large (d >= 0.80)"
    elif abs_d >= 0.50:
        magnitude = "Medium (0.50 <= d < 0.80)"
    elif abs_d >= 0.20:
        magnitude = "Small (0.20 <= d < 0.50)"
    else:
        magnitude = "Negligible (d < 0.20)"

    return {
        "d":         d,
        "abs_d":     abs_d,
        "magnitude": magnitude,
        "paired":    paired,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY INFERENCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(group_a: np.ndarray, group_b: np.ndarray,
                 label_a: str = "Group A", label_b: str = "Group B",
                 alpha: float = 0.05) -> dict:
    """
    Two-tailed paired samples t-test.

    Chapter 3: primary test for RQ1 (trust Codec A vs Codec B) and
    RQ2 (MOS Codec A vs Codec B).
    Significance threshold: alpha = 0.05 (two-tailed).

    Returns full APA-style reporting values.
    """
    t_stat, p_val = scipy_stats.ttest_rel(group_a, group_b)
    df            = len(group_a) - 1
    effect        = cohen_d(group_a, group_b, paired=True)
    diff          = group_a - group_b

    significant = bool(p_val < alpha)
    direction   = f"{label_a} > {label_b}" if np.mean(group_a) > np.mean(group_b) \
                  else f"{label_a} < {label_b}"

    conclusion = (
        f"The paired t-test {'revealed' if significant else 'did not reveal'} a "
        f"statistically {'significant' if significant else 'non-significant'} difference "
        f"between {label_a} (M = {np.mean(group_a):.2f}, SD = {np.std(group_a, ddof=1):.2f}) "
        f"and {label_b} (M = {np.mean(group_b):.2f}, SD = {np.std(group_b, ddof=1):.2f}), "
        f"t({df}) = {t_stat:.3f}, p = {p_val:.4f}, d = {effect['d']:.3f} "
        f"({effect['magnitude']})."
    )

    return {
        "test":         "Paired t-test",
        "label_a":      label_a,
        "label_b":      label_b,
        "n":            len(group_a),
        "df":           df,
        "t_statistic":  float(t_stat),
        "p_value":      float(p_val),
        "significant":  significant,
        "alpha":        alpha,
        "direction":    direction if significant else "No significant difference",
        "mean_a":       float(np.mean(group_a)),
        "sd_a":         float(np.std(group_a, ddof=1)),
        "mean_b":       float(np.mean(group_b)),
        "sd_b":         float(np.std(group_b, ddof=1)),
        "mean_diff":    float(np.mean(diff)),
        "sd_diff":      float(np.std(diff, ddof=1)),
        "cohen_d":      effect,
        "conclusion":   conclusion,
        "apa_string":   f"t({df}) = {t_stat:.3f}, p = {p_val:.4f}, d = {effect['d']:.3f}",
    }


def wilcoxon_signed_rank(group_a: np.ndarray, group_b: np.ndarray,
                          label_a: str = "Group A", label_b: str = "Group B",
                          alpha: float = 0.05) -> dict:
    """
    Wilcoxon signed-rank test — non-parametric paired test.

    Chapter 3: used as primary test if normality is violated,
    or as a robustness check alongside the t-test.
    """
    stat, p_val  = scipy_stats.wilcoxon(group_a, group_b, alternative="two-sided")
    significant  = bool(p_val < alpha)
    n            = len(group_a)

    # Rank-biserial correlation r as effect size
    # r = 1 - (2*W) / (n*(n+1))
    r_effect = float(1 - (2 * stat) / (n * (n + 1)))

    conclusion = (
        f"Wilcoxon signed-rank test {'indicated' if significant else 'did not indicate'} "
        f"a {'significant' if significant else 'non-significant'} difference, "
        f"W = {stat:.1f}, p = {p_val:.4f}, r = {r_effect:.3f}."
    )

    return {
        "test":              "Wilcoxon signed-rank",
        "label_a":           label_a,
        "label_b":           label_b,
        "n":                 n,
        "W_statistic":       float(stat),
        "p_value":           float(p_val),
        "significant":       significant,
        "r_effect_size":     r_effect,
        "conclusion":        conclusion,
        "apa_string":        f"W = {stat:.1f}, p = {p_val:.4f}, r = {r_effect:.3f}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPEATED MEASURES ANOVA  (codec × latency interaction)
# ─────────────────────────────────────────────────────────────────────────────

def repeated_measures_anova(data_dict: dict, alpha: float = 0.05) -> dict:
    """
    2x2 within-subjects (repeated measures) ANOVA using pingouin.

    Tests the codec × latency interaction and both main effects.

    Args:
        data_dict: {"participant": [...], "codec": [...],
                    "latency": [...], "trust_score": [...]}
        Each list has length n_participants * 4 (all combinations).

    Returns dict with main effects, interaction, partial eta-squared.

    Chapter 3: tests whether the effect of explainability on trust
    varies as a function of latency condition.
    """
    try:
        import pingouin as pg
        import pandas as pd

        df = pd.DataFrame(data_dict)

        # 2-way repeated measures ANOVA
        aov = pg.rm_anova(
            data=df,
            dv="trust_score",
            within=["codec", "latency"],
            subject="participant",
            detailed=True,
        )

        results = {}
        for _, row in aov.iterrows():
            source = row["Source"]
            p      = float(row["p_unc"])
            F      = float(row["F"])
            ddof1  = float(row["ddof1"])
            ddof2  = float(row["ddof2"])
            eta2   = float(row.get("ng2", row.get("np2", float("nan"))))

            results[source] = {
                "F":           F,
                "df1":         ddof1,
                "df2":         ddof2,
                "p_value":     p,
                "partial_eta2":eta2,
                "significant": bool(p < alpha),
                "apa_string":  f"F({ddof1:.0f}, {ddof2:.0f}) = {F:.3f}, "
                               f"p = {p:.4f}, partial eta2 = {eta2:.3f}",
            }

        return {
            "test":    "2x2 Repeated Measures ANOVA",
            "factors": ["codec", "latency"],
            "dv":      "trust_score",
            "results": results,
            "library": "pingouin",
        }

    except ImportError:
        # Fallback: separate one-way tests with a note
        return {
            "test":  "Repeated measures ANOVA (pingouin not available)",
            "note":  "Install pingouin for full 2x2 ANOVA. Reporting separate t-tests instead.",
            "results": {},
        }


def eta_squared(ss_effect: float, ss_total: float) -> dict:
    """
    Partial eta-squared effect size for ANOVA.
    Interpretation: small = 0.01, medium = 0.06, large = 0.14
    """
    eta2 = float(ss_effect / ss_total)
    if eta2 >= 0.14:
        magnitude = "Large (eta2 >= 0.14)"
    elif eta2 >= 0.06:
        magnitude = "Medium (0.06 <= eta2 < 0.14)"
    elif eta2 >= 0.01:
        magnitude = "Small (0.01 <= eta2 < 0.06)"
    else:
        magnitude = "Negligible (eta2 < 0.01)"

    return {"eta2": eta2, "magnitude": magnitude}


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def pearson_correlation(x: np.ndarray, y: np.ndarray,
                         label_x: str = "X", label_y: str = "Y",
                         alpha: float = 0.05) -> dict:
    """
    Pearson correlation coefficient.

    Chapter 3: tests the correlation between trust scores (Jian scale)
    and perceived audio quality (MOS ratings).

    Returns r, p, 95% CI, and plain-language interpretation.
    """
    n       = len(x)
    r, p    = scipy_stats.pearsonr(x, y)

    # 95% CI via Fisher's z-transformation
    z       = np.arctanh(r)
    se_z    = 1.0 / np.sqrt(n - 3)
    ci_low  = float(np.tanh(z - 1.96 * se_z))
    ci_high = float(np.tanh(z + 1.96 * se_z))

    abs_r   = abs(r)
    if abs_r >= 0.70:
        strength = "Strong"
    elif abs_r >= 0.50:
        strength = "Moderate"
    elif abs_r >= 0.30:
        strength = "Weak-to-moderate"
    elif abs_r >= 0.10:
        strength = "Weak"
    else:
        strength = "Negligible"

    direction = "positive" if r > 0 else "negative"
    significant = bool(p < alpha)

    conclusion = (
        f"A {strength.lower()}, {direction} correlation was "
        f"{'found' if significant else 'not found'} between {label_x} and {label_y}, "
        f"r({n-2}) = {r:.3f}, p = {p:.4f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]."
    )

    return {
        "r":            float(r),
        "r_squared":    float(r ** 2),
        "p_value":      float(p),
        "df":           n - 2,
        "n":            n,
        "ci95_low":     ci_low,
        "ci95_high":    ci_high,
        "significant":  significant,
        "strength":     strength,
        "direction":    direction,
        "conclusion":   conclusion,
        "apa_string":   f"r({n-2}) = {r:.3f}, p = {p:.4f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]",
    }


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS DECISION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_hypothesis(ttest_result: dict, wilcoxon_result: dict,
                         hypothesis: str, null: str, alternative: str) -> dict:
    """
    Synthesise t-test and Wilcoxon results into a hypothesis decision.
    Both tests must agree to reject the null; if they disagree, report as inconclusive.
    """
    t_sig  = ttest_result["significant"]
    w_sig  = wilcoxon_result["significant"]

    if t_sig and w_sig:
        decision  = "REJECT null hypothesis"
        supported = True
        summary   = f"Both parametric and non-parametric tests are significant. {alternative}"
    elif not t_sig and not w_sig:
        decision  = "FAIL TO REJECT null hypothesis"
        supported = False
        summary   = f"Neither test reached significance. {null}"
    else:
        decision  = "INCONCLUSIVE -- tests disagree"
        supported = None
        summary   = ("Parametric and non-parametric tests give conflicting results. "
                     "Interpret with caution and report both.")

    return {
        "hypothesis":         hypothesis,
        "null":               null,
        "alternative":        alternative,
        "decision":           decision,
        "alternative_supported": supported,
        "summary":            summary,
        "t_significant":      t_sig,
        "wilcoxon_significant": w_sig,
    }
