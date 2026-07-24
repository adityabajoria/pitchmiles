"""
evaluate.py — Rigorous out-of-sample evaluation of the travel-fatigue signal.

The question is NOT "is there an in-sample correlation" (there usually is).
It is: does a travel/rest fatigue signal ADD out-of-sample predictive power
for away-team performance, ON TOP of a strong opponent-strength baseline?

Methodology (the parts that matter for a signal-research audience):
  1. Temporal split — train on early seasons, test on later ones. NEVER random,
     because that leaks future information into the past.
  2. Honest baseline — opponent strength (Elo) alone. Any real signal must beat it
     out-of-sample, not just improve in-sample fit.
  3. Effect size with uncertainty — OLS coefficient on distance with a 95% CI and
     p-value, so we can say whether the effect is distinguishable from zero.
  4. We report where the signal is weak, not just where it looks good.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as st


def temporal_split(df, train_end_year=2021):
    train = df[df.season_start_year <= train_end_year].copy()
    test = df[df.season_start_year > train_end_year].copy()
    return train, test


def fit_eval(train, test, feature_cols):
    scaler = StandardScaler().fit(train[feature_cols])
    Xtr = scaler.transform(train[feature_cols]); ytr = train.away_points.values
    Xte = scaler.transform(test[feature_cols]);  yte = test.away_points.values
    model = Ridge(alpha=1.0).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "mae": mean_absolute_error(yte, pred),
        "r2": r2_score(yte, pred),
        "n_train": len(train), "n_test": len(test),
    }


def mean_baseline(train, test):
    """Dumbest honest baseline: predict the training mean for everyone."""
    pred = np.full(len(test), train.away_points.mean())
    return {
        "mae": mean_absolute_error(test.away_points, pred),
        "r2": r2_score(test.away_points, pred),
    }


def distance_effect(df):
    """OLS of away_points on standardized features; report the distance coef + CI."""
    feats = ["distance_km", "days_rest", "opponent_strength"]
    X = StandardScaler().fit_transform(df[feats])
    X = np.column_stack([np.ones(len(df)), X])
    y = df.away_points.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    names = ["intercept", "distance_km", "days_rest", "opponent_strength"]
    rows = []
    for i, nm in enumerate(names):
        t = beta[i] / se[i]
        p = 2 * (1 - st.t.cdf(abs(t), dof))
        lo, hi = beta[i] - 1.96 * se[i], beta[i] + 1.96 * se[i]
        rows.append((nm, beta[i], se[i], t, p, lo, hi))
    return pd.DataFrame(rows, columns=["feature", "coef", "se", "t", "p_value", "ci_lo", "ci_hi"])


def run(path="epl_features.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} away-match observations "
          f"(seasons {df.season_start_year.min()}-{df.season_start_year.max()})\n")

    train, test = temporal_split(df, train_end_year=2021)
    print(f"Temporal split: train <=2021 ({len(train)} matches), "
          f"test >2021 ({len(test)} matches)\n")

    base = mean_baseline(train, test)
    opp_only = fit_eval(train, test, ["opponent_strength"])
    full = fit_eval(train, test, ["opponent_strength", "distance_km", "days_rest"])

    print("=== Out-of-sample comparison (lower MAE = better) ===")
    print(f"  Mean baseline           : MAE {base['mae']:.4f}   R2 {base['r2']:+.4f}")
    print(f"  Opponent strength only  : MAE {opp_only['mae']:.4f}   R2 {opp_only['r2']:+.4f}")
    print(f"  + travel + rest signal  : MAE {full['mae']:.4f}   R2 {full['r2']:+.4f}")
    lift = opp_only["mae"] - full["mae"]
    print(f"\n  MAE improvement from fatigue features: {lift:+.4f} "
          f"({100*lift/opp_only['mae']:+.2f}%)")
    print("  --> Interpretation: this is how much the travel/rest signal adds")
    print("      OUT-OF-SAMPLE on top of opponent strength. Small/near-zero means")
    print("      the fatigue edge is largely explained by opponent quality.\n")

    print("=== Effect size (full sample OLS, standardized features) ===")
    eff = distance_effect(df)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(eff.to_string(index=False))
    d = eff[eff.feature == "distance_km"].iloc[0]
    sig = "significant" if d.p_value < 0.05 else "NOT significant"
    print(f"\n  Distance effect is {sig} at 5% (p={d.p_value:.3f}). "
          f"95% CI [{d.ci_lo:+.4f}, {d.ci_hi:+.4f}] away-points per SD of distance.")


if __name__ == "__main__":
    run()