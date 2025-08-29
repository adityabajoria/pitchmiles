Research Question: “From 2014 through 2024, how much does each additional 100 km of travel reduce away-team points per game in Brazil’s Série A, Russia’s Premier League, and England’s Premier League, after adjusting for opponent strength and days of rest?”

The focus will be on the countries:
* Premier League (England), Serie A (Brazil), MLS (US)
specifically I will focus on the top 10 clubs from each country.

Premier League (England):
1. Manchester City
2. Liverpool
3. Arsenal
4. Chelsea
5. Manchester United
6. Tottenham Hotspur
7. Newcastle United
8. Leicester City
9. Aston Villa
10. West Ham

Brazilian Serie A (Brazil):
1. Sao Paulo FC
2. Atletico Mineiro
3. Flamengo
4. Palmeiras
5. Corinthians
6. Internacional
7. Santos FC
8. Gremio
9. Botafogo
10. Cruzeiro

MLS (USA):
1. LA Galaxy
2. DC United
3. Houston Dynamo
4. Seattle Sounders
5. Sporting Kansas City
6. Chicago Fire
7. LAFC
8. New York Red Bulls
9. Portland Timbers
10. Philadelphia Union

Step 1: Data Collection --- create a combined csv with the following features:
* date, league, home_team, away_team, home_team_score, away_team_score
* home_latitude, home_longitude, away_latitude, away_longitude, distance_km
* away_points, days_rest, opponent_strength

Step 2: Data Cleaning and Engineering
* Fill missing values, ensure teams team nmaes are consistent across seasons.
* scale features like distance, opponent_strength and rest days.

Step 3: Exploratory Data Analysis (EDA): understand your data, spot patterns, very important for building models.
* Insepect a few rows and basic stats.
* Visualize the spread of each feature.
* Eg. Plot how away_points changes with distance.
* Look at numeric features, helps identify `multicollinearity` before modelling.
* Compare travel vs performance for each league
* Answer the questions, a. Do more rested teams do better with longer travel?
                        b. Do strong opponents cancel out rest benefits?
* Outlier / Anomaly Detection (any match with 10+ rest days. any team with high travel?)
Some important questions to consider:
- What kind of relationship does distance traveled have with away points? (linear/nonlinear)
- is the effect consistent across the different leagues?
- What factors other than rest are more important?
- Do you need to transform any of the variables?
