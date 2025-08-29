import pandas as pd

# Step 1: Load only necessary columns, suppress warnings
use_cols = ['date', 'home', 'away', 'home_score', 'away_score', 'year']
mls_df = pd.read_csv(
    "/Users/aditya/datascience-projects/matches.csv",
    usecols=use_cols,
    parse_dates=['date'],
    low_memory=False
)

# Step 2: Rename to match standard schema
mls_df = mls_df.rename(columns={
    'home': 'home_team',
    'away': 'away_team',
    'year': 'season'
})

# Step 3: Filter to 2014–2024
mls_df = mls_df[mls_df['season'].between(2014, 2024)]

# Step 4: Add league column
mls_df['league'] = 'MLS'

# Step 5: Final column order
mls_df = mls_df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'season', 'league']]

# Step 6: Save to CSV
mls_df.to_csv("/Users/aditya/datascience-projects/soccer-analysis/data/mls_data.csv", index=False)

print("✅ Saved to data/refined_mls_teams.csv")
