# PitchMiles

**Does travel fatigue actually cost teams points on the road, or does opponent strength explain it away?**

Live dashboard: https://pitchmiles.fly.dev

## Research Question

From 2014 to 2024, how much does each additional 100 km of away travel change a team's points per game across England's Premier League, Brazil's Série A, and Major League Soccer, after adjusting for opponent strength and days of rest?

## Why These Three Leagues

Travel distance only becomes a meaningful variable when geography makes it one. England's clubs sit close together, so away trips are short and vary little. Brazil and the United States span thousands of kilometres, producing the wide range of travel distances needed to detect a fatigue effect if one exists. Comparing all three leagues also shows whether any effect scales with distance, which is the core of the research question.

## Approach

The analysis moves from description to formal testing across five sections of the dashboard.

The descriptive views establish the landscape: which teams win, how far each travels, and how competitive balance is distributed within each league. The core view then plots average travel distance against away points, and buckets the most extreme journeys by days of rest to see whether results collapse under the hardest travel.

Because opponent strength is the largest driver of results, the analysis builds an Elo rating for every team and shows that it tracks win percentage closely. Elo is the confounder the study controls for: without it, travel can easily be credited or blamed for results it did not cause.

Finally, a companion out-of-sample model tests the relationship formally, using a temporal train and test split and controlling for pre-match Elo.

## Analysis

**Data pipeline and SQL.** Raw match results for the three leagues begin as CSV files (one per league). These are cleaned and combined, then loaded into a SQLite database (`football.db`), which acts as the single source of truth for the dashboard. The engineered, analysis-ready tables live in SQLite: match-level rows in a `data` table, plus aggregated summaries such as home and away points per team (`home_away_pts`), average travel distance and rest days per team (`avg_distance_restdays`), league standings (`league_rankings`), Elo ratings (`ELO`), and upset records (`UPSETS`). The dashboard reads these tables directly with `SELECT` queries and also runs live SQL at load time, including joins between team-level and match-level tables and aggregations (`AVG`, `GROUP BY`, filtered `WHERE` clauses) to build views like average travel distance against away points. Using SQL for the aggregation keeps the heavy grouping in the database and leaves the Python layer to handle only visualization.

**Feature engineering.** From the raw fixtures, the pipeline derives the variables the research question depends on: travel distance per away fixture (computed from team locations), days of rest between matches, away points earned, and an Elo rating that summarizes opponent strength. Elo is updated match by match, so it reflects each team's form at the time of a given fixture rather than a single season-long average.

**Exploration.** The dashboard works through the question in stages. Descriptive views establish who wins and who travels, then the core view plots average travel against away points and buckets the most extreme journeys by rest. A separate stage builds Elo and demonstrates that it tracks win percentage closely, marking opponent strength as the dominant driver of results and the key variable to control for.

**Key finding.** A companion out-of-sample model tests the relationship formally on the Premier League, using a temporal train and test split and controlling for pre-match Elo. Travel distance is statistically significant in sample, at about -0.05 away points per standard deviation of distance (p = 0.011), but it adds essentially no out-of-sample predictive value once opponent strength is known. The travel-fatigue effect is therefore real but very small, and almost entirely absorbed by opponent quality, so it is not independently useful for predicting away results across these three leagues. The methodological point matters as much as the number: a signal can be genuine in sample and still fail out of sample, which is why the study relies on out-of-sample validation and reports a modest, honest result rather than overstating a weak one.

## Data

Ten seasons (2014 to 2024) of match results from the Premier League, Brazil's Série A, and MLS, focusing on a consistent set of top clubs from each league. Engineered features include travel distance per fixture, days of rest, and a computed Elo rating for opponent strength.

## Tech Stack

Python · Streamlit · pandas · Plotly · SQLite
