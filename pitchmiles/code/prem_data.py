import pandas as pd
import os

import pandas as pd
import os

# Path to your folder
data_dir = "/Users/aditya/datascience-projects/soccer-analysis/prem-data"
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

all_dfs = []

for file in csv_files:
    season_label = file.replace(".csv", "")
    path = os.path.join(data_dir, file)
    df = pd.read_csv(path)

    # Handle column variations
    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_score",
        "FTAG": "away_score"
    })

    # Filter to just the necessary columns
    if all(col in df.columns for col in ["date", "home_team", "away_team", "home_score", "away_score"]):
        df = df[["date", "home_team", "away_team", "home_score", "away_score"]]
        df["season"] = season_label
        df["league"] = "Premier League"
        all_dfs.append(df)

# Merge all
df_all = pd.concat(all_dfs, ignore_index=True)
df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

# Save final merged CSV
df_all.to_csv("epl_2014_2024_combined.csv", index=False)
print("✅ Merged CSV saved as epl_2014_2024_combined.csv")
