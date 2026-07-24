# PitchMiles — Travel-Fatigue Signal Study

An out-of-sample signal-research study: **does a travel/rest fatigue signal add
predictive power for away-team performance, on top of a strong opponent-strength
baseline?** Built on 10 seasons of English Premier League match data (2014–2024).

This is deliberately framed the way an alternative-data research desk evaluates a
candidate signal: form a hypothesis, engineer causal features, test out-of-sample
against an honest baseline, and report the effect size *with its uncertainty* —
including where the signal turns out to be weak.

## Pipeline

1. **`features.py`** — from raw match results (date, teams, scores), engineers:
   - `distance_km` — great-circle travel distance to the venue
   - `days_rest` — days since the away team's previous fixture
   - `opponent_strength` — **causal** pre-match Elo of the home team (no lookahead:
     each match's Elo uses only earlier matches)
   - `away_points` — target (0/1/3)

2. **`evaluate.py`** — the rigorous part:
   - **Temporal split** (train ≤2021, test >2021) — never random, to avoid leaking
     future information into the past.
   - **Honest baselines** — training-mean, then opponent-strength-only. The fatigue
     signal must beat opponent-strength *out-of-sample* to count.
   - **Effect size with 95% CI and p-values** via OLS on standardized features.

## Key finding (EPL, 2014–2024)

| Model (out-of-sample) | MAE | R² |
|---|---|---|
| Mean baseline | 1.170 | −0.000 |
| Opponent strength only | 1.065 | +0.109 |
| + travel + rest signal | 1.065 | +0.112 |

- **In-sample, travel distance is a statistically significant drag** on away points
  (≈ −0.05 pts per SD of distance; 95% CI [−0.09, −0.01]; p = 0.011).
- **Out-of-sample, the fatigue signal adds ~0% incremental accuracy** once opponent
  strength is known — its apparent edge is largely absorbed by opponent quality.

**Interpretation.** The fatigue effect is real but not *independently* predictive at
the match level once you control for the obvious factor. This is the common life cycle
of a candidate signal, and the honest reporting of it — rather than an inflated
in-sample story — is the point of the study.

## Run it

```bash
pip install pandas numpy scikit-learn scipy
python features.py     # writes epl_features.csv
python evaluate.py     # prints the comparison + effect sizes
```

## Extending to other leagues

`features.py` includes EPL venue coordinates. Brasileirão and MLS raw files are
included in the parent repo; add their club coordinates to the `COORDS` map and
normalize their date/column formats to reuse the same evaluation pipeline.
