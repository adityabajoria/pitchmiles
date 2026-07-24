"""
features.py — Feature engineering for the PitchMiles travel-fatigue signal study.

Builds, from raw match results (date, teams, scores), a per-away-match dataset with:
  - distance_km      : great-circle distance from away team's home city to match venue
  - days_rest        : days since the away team's previous match
  - opponent_strength: pre-match Elo rating of the home team (no lookahead)
  - away_points      : target — points won by the away team (0/1/3)

Design notes (these matter for the methodology):
  - Elo is computed causally, iterating matches in date order. A match's
    opponent_strength uses only information available BEFORE that match, so there
    is no target leakage.
  - days_rest is computed per team from their own prior fixture.
  - All features are known at kickoff, so the resulting dataset is a fair
    out-of-sample prediction setup.
"""

import numpy as np
import pandas as pd

# --- EPL home city coordinates (lat, lon). Approx city/stadium locations. ---
# Enough to compute realistic travel distances across 2014-2024 EPL clubs.
COORDS = {
    "Arsenal": (51.5549, -0.1084), "Aston Villa": (52.5092, -1.8848),
    "Bournemouth": (50.7352, -1.8383), "Brentford": (51.4907, -0.2887),
    "Brighton": (50.8616, -0.0837), "Burnley": (53.7890, -2.2300),
    "Cardiff": (51.4728, -3.2030), "Chelsea": (51.4817, -0.1910),
    "Crystal Palace": (51.3983, -0.0857), "Everton": (53.4388, -2.9663),
    "Fulham": (51.4749, -0.2217), "Huddersfield": (53.6540, -1.7684),
    "Hull": (53.7460, -0.3676), "Leeds": (53.7778, -1.5722),
    "Leicester": (52.6203, -1.1422), "Liverpool": (53.4308, -2.9608),
    "Luton": (51.8842, -0.4316), "Man City": (53.4831, -2.2004),
    "Man United": (53.4631, -2.2913), "Middlesbrough": (54.5780, -1.2169),
    "Newcastle": (54.9756, -1.6217), "Norwich": (52.6220, 1.3092),
    "Nott'm Forest": (52.9400, -1.1327), "QPR": (51.5093, -0.2320),
    "Sheffield United": (53.3703, -1.4709), "Southampton": (50.9058, -1.3910),
    "Stoke": (52.9884, -2.1755), "Sunderland": (54.9145, -1.3882),
    "Swansea": (51.6428, -3.9351), "Tottenham": (51.6043, -0.0665),
    "Watford": (51.6499, -0.4015), "West Brom": (52.5090, -1.9640),
    "West Ham": (51.5387, -0.0166), "Wolves": (52.5902, -2.1300),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_epl(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Keep only teams we have coordinates for (all EPL clubs in this window)
    df = df[df["home_team"].isin(COORDS) & df["away_team"].isin(COORDS)].copy()
    return df


def compute_elo(df, k=20, base=1500.0, home_adv=60.0):
    """Causal Elo: ratings reflect only matches strictly before the current one."""
    ratings = {t: base for t in pd.unique(df[["home_team", "away_team"]].values.ravel())}
    home_elos, away_elos = [], []
    for _, row in df.iterrows():
        rh, ra = ratings[row.home_team], ratings[row.away_team]
        home_elos.append(rh)   # pre-match rating -> no lookahead
        away_elos.append(ra)
        # expected home result including home advantage
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + home_adv) - ra) / 400))
        if row.home_score > row.away_score:
            s_h = 1.0
        elif row.home_score < row.away_score:
            s_h = 0.0
        else:
            s_h = 0.5
        ratings[row.home_team] = rh + k * (s_h - exp_h)
        ratings[row.away_team] = ra + k * ((1 - s_h) - (1 - exp_h))
    df = df.copy()
    df["home_elo_pre"] = home_elos
    df["away_elo_pre"] = away_elos
    return df


def build_away_dataset(df):
    """One row per match, from the away team's perspective."""
    df = compute_elo(df)

    # days_rest per team (min over home/away appearances)
    last_played = {}
    rest = []
    for _, row in df.iterrows():
        d = row.date
        ar = last_played.get(row.away_team)
        rest.append((d - ar).days if ar is not None else np.nan)
        last_played[row.home_team] = d
        last_played[row.away_team] = d
    df["days_rest"] = rest

    # travel distance: away city -> home (venue) city
    def dist(row):
        la = COORDS[row.away_team]; lh = COORDS[row.home_team]
        return haversine_km(la[0], la[1], lh[0], lh[1])
    df["distance_km"] = df.apply(dist, axis=1)

    # opponent strength = home team's pre-match Elo (the away team's opponent)
    df["opponent_strength"] = df["home_elo_pre"]
    df["own_strength"] = df["away_elo_pre"]

    # target: away points
    def away_pts(row):
        if row.away_score > row.home_score: return 3
        if row.away_score == row.home_score: return 1
        return 0
    df["away_points"] = df.apply(away_pts, axis=1)

    out = df.dropna(subset=["days_rest", "distance_km", "opponent_strength"]).copy()
    out["season_start_year"] = out["date"].dt.year
    return out[[
        "date", "season", "season_start_year", "home_team", "away_team",
        "distance_km", "days_rest", "opponent_strength", "own_strength", "away_points",
    ]].reset_index(drop=True)


if __name__ == "__main__":
    raw = load_epl("../epl_2014_2024_combined.csv")
    ds = build_away_dataset(raw)
    ds.to_csv("epl_features.csv", index=False)
    print(f"Built {len(ds)} away-match rows, {ds.date.min().date()} -> {ds.date.max().date()}")
    print(ds[["distance_km", "days_rest", "opponent_strength", "away_points"]].describe().round(1))