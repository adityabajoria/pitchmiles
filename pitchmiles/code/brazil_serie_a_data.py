import pandas as pd

# Load the dataset
brazil_df = pd.read_csv('/Users/aditya/datascience-projects/soccer-analysis/data/Brasileirao_Matches.csv', parse_dates=['datetime'])

# Rename columns to standard format
brazil_df = brazil_df.rename(columns={
    'datetime': 'date',
    'home_team': 'home_team',
    'away_team': 'away_team',
    'home_goal': 'home_score',
    'away_goal': 'away_score',
    'season': 'season'
})

# Filter by season only (no 'serie' column needed)
brazil_df = brazil_df[brazil_df['season'].between(2014, 2024)]

# Add league column
brazil_df['league'] = 'Brazilian Serie A'

# Optional: Select only the columns you need
brazil_df = brazil_df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'season', 'league']]

# Save to file
brazil_df.to_csv('brazil_2014_2024_combined.csv', index=False)

print("✅ Brazilian Serie A data cleaned and saved.")