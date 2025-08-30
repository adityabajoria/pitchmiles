import requests
import os
import numpy as np
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from pathlib import Path
from itertools import islice

DATASETTE_URL = "https://pitchmiles-datasette.fly.dev/football"

DB_PATH = Path(__file__).resolve().parents[1] / "sql" / "football.db"
conn = sqlite3.connect(DB_PATH)

# --- GLOBAL STYLING ---
st.markdown(
    """
    <style>
    /* --- GLOBAL APP BACKGROUND --- */
    .stApp {
        background-color: #0a192f !important;  /* navy */
        color: #ffffff !important;             /* white text */
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a192f !important;
        color: #ffffff !important;
    }

    /* Force all text white */
    html, body, .stApp, .stMarkdown, [class*="st-"],
    [data-testid="stMarkdownContainer"], label {
        color: #ffffff !important;
    }

    /* --- TABLE STYLING --- */
    thead tr th {
        background-color: #0a192f !important;  /* navy header */
        color: #ffffff !important;             /* white text in header */
    }
    tbody tr td {
        background-color: #ffffff !important;  /* white rows */
        color: #0a192f !important;             /* navy text */
    }

    /* --- SELECTBOX CLOSED STATE --- */
    div[data-baseweb="select"] > div {
        background-color: #0a192f !important;  /* navy */
        color: #ffffff !important;             /* white text */
        border-radius: 6px;
        border: 1px solid #1e3a8a !important;  /* subtle border */
    }

    /* --- SELECTBOX DROPDOWN MENU --- */
    ul[role="listbox"] {
        background-color: #000000 !important;  /* black dropdown background */
        color: #ffffff !important;             /* white text */
        border-radius: 6px;
    }

    /* Dropdown options */
    ul[role="listbox"] li {
        background-color: #000000 !important;  /* black options */
        color: #ffffff !important;             /* white text */
    }
    ul[role="listbox"] li:hover {
        background-color: #1e3a8a !important;  /* navy hover highlight */
    }

    /* Dropdown arrow icon */
    div[data-baseweb="select"] svg {
        fill: #ffffff !important;  /* white arrow */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- NAVBAR ---
selected = option_menu(
    menu_title=None,
    options=["Home", "Overview", "Travel & Performance", "Upsets & Opponent Strength", "Team Comparison"],
    icons=["house", "bar-chart", "people", "trophy", "shuffle"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0a192f"},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "color": "white", "padding": "10px"},
        "nav-link-selected": {"background-color": "#1e3a8a", "color": "white"},
    }
)

# --- DASHBOARD HEADER (Always at top for all pages) ---
st.markdown(f"<h1 style='text-align: center;'>⚽ PitchMiles – {selected}</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #3A9BDC;'>", unsafe_allow_html=True)

if selected == "Home":
    st.write("**Rearch Question**: How does travel distance affect away-team performance across the **3 geographically different leagues** between the **2014-2024?**")
    st.markdown(
    """
    <p style='font-size:17px; font-weight:600;'>
        Leagues: Premier League (England), Série A (Brazil), Major League Soccer (USA)
    </p>
    """,
    unsafe_allow_html=True
)
    matches_played = 1140
    goals_scored = 2985
    season_range = "2014-2024"
    with st.container():
        st.caption("KEY OVERVIEW")
        c1, c2, c3 = st.columns(3)
        c1.metric("Matches Played", f"{matches_played}")
        c2.metric("Goals Scored", f"{goals_scored}")
        c3.metric("Season", season_range)

    map_data = pd.DataFrame({
        "League": ["EPL", "MLS", "Brazilian League"],
        "Latitude": [51.509865, 37.0902, -14.2350],
        "Longitude": [-0.118092, -95.7129, -51.9253]
    })

    fig = px.fig = px.scatter_mapbox(
        map_data,
        lat="Latitude",
        lon="Longitude",
        hover_name="League",
        color_discrete_sequence=["red"],
        zoom=0.5,
        height=400
    )

    fig.update_layout(
        mapbox_style = "carto-darkmatter",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- DATA ---------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGO_DIR = os.path.join(BASE_DIR, "..", "logos")  # adjust if needed

    # --- CLUB DATA --------------------------------------------------------
    premier = [
        ("Man City", f"{LOGO_DIR}/prem/Manchester_City_FC_badge.svg"),
        ("Liverpool", f"{LOGO_DIR}/prem/Liverpool_FC.svg"),
        ("Arsenal", f"{LOGO_DIR}/prem/Arsenal_FC.svg"),
        ("Chelsea", f"{LOGO_DIR}/prem/Chelsea_FC.svg"),
        ("Man United", f"{LOGO_DIR}/prem/Manchester_United_FC_crest.svg"),
        ("Tottenham", f"{LOGO_DIR}/prem/Tottenham_Hotspur.svg"),
        ("Newcastle", f"{LOGO_DIR}/prem/Newcastle_United_Logo.svg"),
        ("Leicester City", f"{LOGO_DIR}/prem/Leicester_City_crest.svg"),
        ("Aston Villa", f"{LOGO_DIR}/prem/Aston_Villa_FC_new_crest.svg"),
        ("West Ham", f"{LOGO_DIR}/prem/West_Ham_United_FC_logo.svg"),
    ]

    brazil = [
        ("Sao Paulo FC", f"{LOGO_DIR}/brazilian/Brasao_do_Sao_Paulo_Futebol_Clube.svg"),
        ("Atletico Mineiro", f"{LOGO_DIR}/brazilian/Clube_Atlético_Mineiro_crest.svg"),
        ("Flamengo", f"{LOGO_DIR}/brazilian/Flamengo_braz_logo.svg"),
        ("Palmeiras", f"{LOGO_DIR}/brazilian/Palmeiras_logo.svg"),
        ("Corinthians", f"{LOGO_DIR}/brazilian/Sport_Club_Corinthians_Paulista_Logo.png"),
        ("Internacional", f"{LOGO_DIR}/brazilian/SC_Internacional_Brazil_Logo.svg"),
        ("Santos FC", f"{LOGO_DIR}/brazilian/Santos_Logo.png"),
        ("Gremio", f"{LOGO_DIR}/brazilian/Gremio_logo.svg"),
        ("Botafogo", f"{LOGO_DIR}/brazilian/Botafogo_de_Futebol_e_Regatas_logo.svg"),
        ("Cruzeiro", f"{LOGO_DIR}/brazilian/Cruzeiro_Esporte_Clube_(logo).svg"),
    ]

    mls = [
        ("LA Galaxy", f"{LOGO_DIR}/mls/Los_Angeles_Galaxy_logo.svg"),
        ("DC United", f"{LOGO_DIR}/mls/D.C._United_logo_(2016).svg"),
        ("Houston Dynamo", f"{LOGO_DIR}/mls/Houston_Dynamo_FC_logo.svg"),
        ("Seattle Sounders", f"{LOGO_DIR}/mls/Seattle_Sounders_logo.svg"),
        ("Sporting Kansas City", f"{LOGO_DIR}/mls/Sporting_Kansas_City_logo.svg"),
        ("Chicago Fire", f"{LOGO_DIR}/mls/Chicago_Fire_logo,_2021.svg"),
        ("LAFC", f"{LOGO_DIR}/mls/Los_Angeles_Football_Club.svg"),
        ("New York Red Bulls", f"{LOGO_DIR}/mls/New_York_Red_Bulls_logo.svg"),
        ("Portland Timbers", f"{LOGO_DIR}/mls/Portland_Timbers_logo.svg"),
        ("Philadelphia Union", f"{LOGO_DIR}/mls/Philadelphia_Union_2018_logo.svg"),
    ]

    LEAGUES = [
        ("Premier League", premier),
        ("Brazilian League", brazil),
        ("MLS League", mls),
    ]

    # --- STYLES -----------------------------------------------------------
    st.markdown("""
    <style>
    .club-cell { text-align:center; margin-bottom:18px; }
    .club-name {
    display:block; margin-top:6px; font-size:14px; line-height:1.1;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    max-width:110px; margin-left:auto; margin-right:auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- HELPERS ----------------------------------------------------------
    def chunks(seq, n):
        it = iter(seq)
        while True:
            chunk = list(islice(it, n))
            if not chunk:
                return
            yield chunk

    def league_grid(title, clubs, per_row=6, logo_size=80):
        st.subheader(title)
        for row in chunks(clubs, per_row):
            cols = st.columns(len(row))
            for col, (name, path) in zip(cols, row):
                with col:
                    st.markdown('<div class="club-cell">', unsafe_allow_html=True)
                    st.image(path, width=logo_size)
                    st.markdown(f"<span class='club-name'>{name}</span>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE --------------------------------------------------------------
    st.header("CLUBS BY League")
    for league_name, club_list in LEAGUES:
        league_grid(league_name, club_list, per_row=6, logo_size=82)


elif selected == "Overview":
    tables = ["overview", "league_rankings", "home_away_pts"]
    selected = st.selectbox("Select a table to view:", tables)
    url = f"{DATASETTE_URL}/{selected}.json?_shape=array&_size=max"
    sql_tables = pd.DataFrame(requests.get(url).json())
    st.dataframe(sql_tables, use_container_width=True)
    st.markdown(f"[Open in Datasette] ({DATASETTE_URL}/{selected})")

    st.subheader("Rank Progression Across Seasons")
    overview_url = f"{DATASETTE_URL}/overview.json?_shape=array&_size=max"
    league_rankings_url = f"{DATASETTE_URL}/league_rankings.json?_shape=array&_size=max"
    overview_df = pd.DataFrame(requests.get(overview_url).json())
    rankings_df = pd.DataFrame(requests.get(league_rankings_url).json())
    merged_df = overview_df.merge(rankings_df[['team', 'league', 'win_percentage']], on=['team', 'league'], how='left')
    merged_df['league_ranks'] = merged_df.groupby(['league', 'season'])['win_percentage'].rank(ascending=False, method='dense')
    selected_league = st.selectbox("Select League:", merged_df['league'].unique())
    selected_season = st.selectbox("Select Season:", merged_df['season'].unique())
    filtered = merged_df[(merged_df['league'] == selected_league) & (merged_df['season'] == selected_season)]
    fig = px.pie(
        filtered,
        names='team',
        values='win_percentage',
        color='team',
        title=f"{selected_league}, {selected_season}",
        hole=0.4
    )
    fig.update_traces(textinfo='percent+label')  # Show percentages & labels
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Average Travel Distance By Team")
    travel_url = f"{DATASETTE_URL}/avg_distance_restdays.json?_shape=array&_size=max"
    travel_df = pd.DataFrame(requests.get(travel_url).json())
    fig = px.bar(
        travel_df,
        x="avg_distance",
        y="team",
        color="league",
        orientation="h",
        title="Average Travel Distance by Team",
        labels={"avg_distance": "Avg Distance (km)", "team": "Team"},
        height=500
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


elif selected == "Travel & Performance":
    tab1, tab2 = st.tabs(["Travel Tiers", "Extreme Travel"])

    with tab1:
        # --- TAB 1 ---
        tables = ["travel_tiers", "travel_pts_bin"]
        selected = st.selectbox("Select a table to view:", tables, key="t1_tbl")
        url = f"{DATASETTE_URL}/{selected}.json?_shape=array&_size=max"

        try:
            sql_tables = pd.DataFrame(requests.get(url).json())
            st.dataframe(sql_tables, use_container_width=True)
            st.markdown(f"[Open in Datasette]({DATASETTE_URL}/{selected})")
        except Exception as e:
            st.warning(f"Could not load table: {e}")

        # Scatter: Average Travel Distance vs Away Points Per Team
        query = """
        SELECT h.league, h.team,
            CAST(h.away_points AS FLOAT) AS away_points,
            d.avg_travel_km
        FROM home_away_pts AS h
        JOIN (
            SELECT league, away_team AS team, AVG(distance_km) AS avg_travel_km
            FROM data
            WHERE distance_km IS NOT NULL
            GROUP BY league, away_team
        ) AS d
        ON h.league = d.league AND h.team = d.team
        WHERE h.away_points IS NOT NULL
        """
        distance_away_pts = pd.read_sql(query, conn).dropna(subset=["avg_travel_km","away_points"])

        if distance_away_pts.empty:
            st.info("No data found for the scatter plot.")
        else:
            y_max = distance_away_pts['away_points'].max()
            fig = px.scatter(
                distance_away_pts,
                x="avg_travel_km",
                y="away_points",
                color="league",
                hover_name="team",
                labels={"avg_travel_km":"Avg Travel Distance (km)", "away_points":"Away Points"},
                title="Average Travel Distance vs Away Points Per Team",
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(255,255,255,0.6)")))
            fig.update_yaxes(range=[0, y_max + 20], rangemode="tozero", showgrid=True, gridwidth=0.3)
            fig.update_xaxes(showgrid=True, gridwidth=0.3)
            fig.update_layout(
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


    with tab2:
        # --- TAB 2 ---
        tables = ["extreme_travel", "fatigue_loss"]
        selected = st.selectbox("Select a table to view:", tables, key="t2_tbl")
        url = f"{DATASETTE_URL}/{selected}.json?_shape=array&_size=max"

        try:
            sql_tables = pd.DataFrame(requests.get(url).json())
            st.dataframe(sql_tables, use_container_width=True)
            st.markdown(f"[Open in Datasette]({DATASETTE_URL}/{selected})")
        except Exception as e:
            st.warning(f"Could not load table: {e}")

        # Load extreme travel rows (no averages/means used)
        extreme_url = f"{DATASETTE_URL}/extreme_travel.json?_shape=array&_size=max"
        extreme_data = pd.DataFrame(requests.get(extreme_url).json())

        # numeric safety
        extreme_data["away_points_earned"] = pd.to_numeric(extreme_data.get("away_points_earned"), errors="coerce")
        extreme_data["days_rest"] = pd.to_numeric(extreme_data.get("days_rest"), errors="coerce")

        # bin days of rest (adjust edges/labels if you prefer)
        rest_bins   = [-0.1, 50, 100, 150, 200, float("inf")]
        rest_labels = ["0–50", "51–100", "101–150", "151–200", "201+"]
        extreme_data["rest_bin"] = pd.cut(extreme_data["days_rest"], bins=rest_bins, labels=rest_labels, include_lowest=True)

        # points categories (0,1,3 only)
        valid_pts = {0,1,3}
        extreme_data["points_cat"] = extreme_data["away_points_earned"].astype("Int64")
        extreme_data = extreme_data[extreme_data["points_cat"].isin(valid_pts)]
        extreme_data = extreme_data.dropna(subset=["rest_bin"])

        # count matches per (rest_bin, points_cat)
        counts = (
            extreme_data
            .groupby(["rest_bin","points_cat"])
            .size()
            .reset_index(name="n")
        )

        # ensure nice order
        counts["rest_bin"] = pd.Categorical(counts["rest_bin"], categories=rest_labels, ordered=True)
        counts = counts.sort_values(["rest_bin","points_cat"]).reset_index(drop=True)

        st.subheader("Extreme Travel — Away Points by Days of Rest")
        fig = px.bar(
            counts,
            x="rest_bin",
            y="n",
            color="points_cat",
            barmode="stack",
            text="n",
            labels={"rest_bin":"Days of Rest (bins)", "n":"Matches", "points_cat":"Away Points"},
            title="Distribution of Away Points by Rest Days (Extreme Travel Matches)",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=520,
            legend_title_text="Away Points",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show underlying counts"):
            st.dataframe(counts, use_container_width=True, hide_index=True)


elif selected == "Upsets & Opponent Strength":
    tab1, tab2, tab3, = st.tabs(['ELO', 'ELO vs Win Percentage', 'UPSETS'])
    # Compute ELO
    def compute_elo(df, k=20, base_rating=1000):
        """
        Compute Elo ratings for each team over time.
        df: DataFrame with ['date', 'home_team', 'away_team', 'home_score', 'away_score']
        Returns: DataFrame with additional Elo columns
        """
        # Ensure date is datetime and sort
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Initialize ratings
        teams = pd.concat([df['home_team'], df['away_team']]).unique()
        ratings = {team: base_rating for team in teams}

        # Track Elo history
        home_elos, away_elos = [], []

        for _, row in df.iterrows():
            home, away = row['home_team'], row['away_team']
            R_home, R_away = ratings[home], ratings[away]

            # Expected results
            E_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
            E_away = 1 - E_home

            # Actual results
            if row['home_score'] > row['away_score']:
                S_home, S_away = 1, 0
            elif row['home_score'] < row['away_score']:
                S_home, S_away = 0, 1
            else:
                S_home, S_away = 0.5, 0.5

            # Update ratings
            ratings[home] = R_home + k * (S_home - E_home)
            ratings[away] = R_away + k * (S_away - E_away)

            # Save post-match ratings
            home_elos.append(ratings[home])
            away_elos.append(ratings[away])

        df['home_elo'] = home_elos
        df['away_elo'] = away_elos

        return df, ratings

    matches = pd.read_sql(
        '''SELECT league, date, home_team, away_team, home_score, away_score FROM data;''',
    conn)

    matches_with_elo, final_ratings = compute_elo(matches)

    elo_table = pd.DataFrame(final_ratings.items(), columns=['Team', 'Elo'])
    elo_table = elo_table.sort_values('Elo', ascending=False).reset_index(drop=True)

    with tab1:
        st.subheader ("What is ELO")
        st.write("The ELO Rating System is a way to numerically measure the relative skill of teams based on their match results. It's a way to compare the strength of the teams.")
        st.markdown("""
        - Initially, every team has a rating of 1000.
        - After a match, points are updated between teams based on the result.
        - Essentially, beating a stronger opponent gives more points than beating a weaker one.
        - Losing to a much weaker opponent costs you more points than losing to an equally strong team.
        """)

        st.subheader("How is ELO Scored")
        st.write("The ELO Score is calculated using a standard formula.")
        st.write("The updated rating for any 'Team A' after a match will be:")
        st.markdown("""
        **$R_{A_{new}} = R_{A_{old}} + K \\cdot (S_A - E_A)$**
        """)
        st.write("WHERE")
        st.markdown("""
        - **$R_{A_{new}}$**: Team A's new rating
        - **$R_{A_{old}}$**: Team A's old rating
        - **$K$**: Weight (how much ratings change per game) – K=20
        - **$S_A$**: Actual score (1 = win, 0.5 = draw, 0 = loss)
        - **$E_A$**: Expected score (calculated using its own formula)
        """)
        st.subheader("ELO Ratings Leaderboard")
        st.dataframe(elo_table, use_container_width=True)


        elo_long = pd.concat([
        matches_with_elo[['date', 'league', 'home_team', 'home_elo']].rename(
            columns={'home_team': 'team', 'home_elo': 'elo'}),
        matches_with_elo[['date', 'league', 'away_team', 'away_elo']].rename(
            columns={'away_team': 'team', 'away_elo': 'elo'})
        ])
        elo_long = elo_long.sort_values(['team', 'date'])
        st.subheader("Elo Ratings Over Time")
        # League & team filters
        selected_league = st.selectbox("Select League", sorted(elo_long['league'].unique()), index=None, placeholder="Choose a league")
        teams_in_league = sorted(elo_long[elo_long['league'] == selected_league]['team'].unique())
        selected_team = st.selectbox("Select Team", teams_in_league, index=None, placeholder="Choose a team")

        # Filter data
        filtered_elo = elo_long[(elo_long['league'] == selected_league) & (elo_long['team'] == selected_team)]

        # Plot
        fig = px.line(
            filtered_elo,
            x='date',
            y='elo',
            title=f"Elo Rating Over Time: {selected_team} ({selected_league})",
            labels={'elo': 'Elo Rating', 'date': 'Date'},
            template="plotly_dark"
        )

        fig.update_traces(line=dict(width=1))
        fig.update_layout(height=500, width=900)

        st.plotly_chart(fig, use_container_width=True)


    # ELO and Home Win Percentage (Scatter)
    query = '''
    SElECT e.ELO, u.home_team, u.home_win_percentage
    FROM ELO e
    JOIN UPSETS u ON e.team = u.home_team
    '''
    home = pd.read_sql_query(query, conn)

    # ELO and Away Win Percentage (Scatter)
    query = '''
    SELECT e.Elo, u.away_team, u.away_win_percentage
    FROM ELO e
    JOIN UPSETS u ON e.team = u.away_team;
    '''
    away = pd.read_sql_query(query, conn)

    with tab2:
        st.subheader("ELO VS Home Win Percentage")
        max_y = home['home_win_percentage'].max()
        fig_home = px.scatter(
            home,
            x="Elo",
            y="home_win_percentage",
            color="home_team",
            hover_data=["home_team","Elo", "home_win_percentage"],
            title="ELO vs Home Win Percentage",
            labels={"home_team": "Team","ELO_pts": "ELO Rating", "home_win_percentage": "Home Win Percentage"},
            template="plotly_dark"
        )
        fig_home.update_yaxes(range=[0, max_y + 10])
        fig_home.update_layout(
            width=1000,
            height=600
        )
        st.plotly_chart(fig_home, use_container_width=True)

        st.subheader("ELO VS Away Win Percentage")
        max_y = away['away_win_percentage'].max()
        fig_away = px.scatter(
            away,
            x="Elo",
            y="away_win_percentage",
            color="away_team",
            hover_data=["away_team", "Elo", "away_win_percentage"],
            title="ELO vs Away Win Percentage",
            labels={"away_team": "Team", "ELO_pts": "ELO Rating", "away_win_percentage": "Away Win Percentage"},
            template="plotly_dark"
        )
        fig_away.update_yaxes(range=[0, max_y + 10])
        fig_away.update_layout(
            width=1000,
            height=600
        )
        st.plotly_chart(fig_away, use_container_width=True)

    # UPSET MATCHES
    with tab3:
        upset_url = f"{DATASETTE_URL}/UPSETS.json?_shape=array&_size=max"
        sql_tables = pd.DataFrame(requests.get(upset_url).json())
        season_counts = sql_tables.groupby("season").size().reset_index(name="Upset_Count")
        season_counts = season_counts.sort_values("season")

        max_y = season_counts['Upset_Count'].max()
        fig_upsets = px.line(
            season_counts,
            x="season",
            y="Upset_Count",
            markers=True,
            title="Upsets Per Team"
        )
        fig_upsets.update_yaxes(range=[0, max_y + 5])
        fig_upsets.update_layout(xaxis_title="Season", yaxis_title="Number of Upsets")
        st.plotly_chart(fig_upsets, use_container_width=True)

        st.subheader("**UPSET Matches**")
        # Rename columns for display
        sql_tables.rename(columns={
            "rowid": "ID",
            "league": "League",
            "season": "Season",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "home_score": "Home Score",
            "home_win_percentage": "Home Win %",
            "away_win_percentage": "Away Win %"
        }, inplace=True)

        # Reorder columns for readability
        sql_tables = sql_tables[["League", "Season", "Home Team", "Away Team", "Home Score", "Home Win %", "Away Win %"]]

        st.dataframe(sql_tables, use_container_width=True)


else:
    st.write("Compare 3 different teams across leagues")
    rankings = pd.DataFrame(requests.get(f"{DATASETTE_URL}/league_rankings.json?_shape=array&_size=max").json())
    travel = pd.DataFrame(requests.get(f"{DATASETTE_URL}/avg_distance_restdays.json?_shape=array&_size=max").json())
    home_away = pd.DataFrame(requests.get(f"{DATASETTE_URL}/home_away_pts.json?_shape=array&_size=max").json())

    # Merge them
    comparison = (
        rankings
        .merge(travel, on=["league","team"], how="left")
        .merge(home_away[["league","team","home_points","away_points","total_points"]], on=["league","team"], how="left")
    )

    st.subheader("Team Comparison (Cross-League)")

    team1 = st.selectbox(
        "Select Team 1",
        [f"{row['team']} ({row['league']})" for _, row in comparison.iterrows()],
        index=0
    )
    team2 = st.selectbox(
        "Select Team 2",
        [f"{row['team']} ({row['league']})" for _, row in comparison.iterrows()],
        index=1
    )
    team3 = st.selectbox(
        "Select Team 3",
        [f"{row['team']} ({row['league']})" for _, row in comparison.iterrows()],
        index=1
    )

    # Extract just team names
    team1_name = team1.split(" (")[0]
    team2_name = team2.split(" (")[0]
    team3_name = team3.split(" (")[0]

    team1_data = comparison[comparison['team'] == team1_name].iloc[0]
    team2_data = comparison[comparison['team'] == team2_name].iloc[0]
    team3_data = comparison[comparison['team'] == team3_name].iloc[0]

    st.markdown(f'**{team1}** vs **{team2}** vs **{team3}**')

    col1, col2, col3 = st.columns(3)

    if len(set([team1, team2, team3])) < 3:
        st.error("Please select 3 different teams for comparision.")
    else:
        with col1:
            st.write(f'{team1}')
            st.metric("Win %", f"{team1_data['win_percentage']:.1f}%")
            st.metric("Avg Travel", f"{team1_data['avg_distance']:.1f} km")
            st.metric("Avg Rest Days", f"{team1_data['avg_restdays']:.1f} days")
            st.metric("Away Points", f"{team1_data['away_points']:.1f}")
        with col2:
            st.write(f'{team2}')
            st.metric("Win %", f"{team2_data['win_percentage']:.1f}%")
            st.metric("Avg Travel", f"{team2_data['avg_distance']:.1f} km")
            st.metric("Avg Rest Days", f"{team2_data['avg_restdays']:.1f} days")
            st.metric("Away Points", f"{team2_data['away_points']:.1f}")
        with col3:
            st.write(f'{team3}')
            st.metric("Win %", f"{team3_data['win_percentage']:.1f}%")
            st.metric("Avg Travel", f"{team3_data['avg_distance']:.1f} km")
            st.metric("Avg Rest Days", f"{team3_data['avg_restdays']:.1f} days")
            st.metric("Away Points", f"{team3_data['away_points']:.1f}")
