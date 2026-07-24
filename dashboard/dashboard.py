import os
import sqlite3
from pathlib import Path
from itertools import islice

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="PitchMiles", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------------------
# DATABASE — everything reads from the local SQLite file (no external service)
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parents[1] / "sql" / "football.db"


@st.cache_resource
def get_conn():
    # check_same_thread=False so Streamlit's rerun threads can share the conn
    return sqlite3.connect(DB_PATH, check_same_thread=False)


conn = get_conn()


def load_table(name):
    """Read an entire table from the local DB. Replaces the old Datasette calls."""
    try:
        return pd.read_sql(f'SELECT * FROM "{name}"', conn)
    except Exception as e:
        st.warning(f"Could not load table '{name}': {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# GLOBAL STYLING  (professional navy theme + gradient header + styled nav)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Base app */
    .stApp {
        background-color: #0a192f !important;
        color: #e6f1ff !important;
    }
    [data-testid="stSidebar"] { background-color: #0a192f !important; color: #e6f1ff !important; }
    html, body, .stApp, .stMarkdown, [class*="st-"],
    [data-testid="stMarkdownContainer"], label { color: #e6f1ff !important; }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- Gradient hero header --- */
    .pm-hero {
        background: linear-gradient(135deg, #0a192f 0%, #112d4e 45%, #1e3a8a 100%);
        border: 1px solid #1e3a8a;
        border-radius: 16px;
        padding: 26px 32px;
        margin: 4px 0 10px 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .pm-hero h1 {
        margin: 0;
        font-size: 40px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .pm-hero .pm-sub {
        margin-top: 6px;
        font-size: 15px;
        font-weight: 500;
        color: #9ecbff !important;
    }
    .pm-accent {
        height: 4px;
        width: 120px;
        margin-top: 14px;
        border-radius: 2px;
        background: linear-gradient(90deg, #3A9BDC, #64ffda);
    }

    /* --- Radio nav styled as pill tabs --- */
    div[role="radiogroup"] {
        gap: 8px;
        flex-wrap: wrap;
        background: #0d2140;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid #16345c;
    }
    div[role="radiogroup"] label {
        background: transparent;
        padding: 8px 16px;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.15s ease;
    }
    div[role="radiogroup"] label:hover { background: #16345c; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0d2140;
        border: 1px solid #16345c;
        border-radius: 12px;
        padding: 14px 16px;
    }

    /* Tables */
    thead tr th { background-color: #0d2140 !important; color: #e6f1ff !important; }
    tbody tr td { background-color: #ffffff !important; color: #0a192f !important; }

    /* Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #0d2140 !important; color: #e6f1ff !important;
        border-radius: 8px; border: 1px solid #1e3a8a !important;
    }
    ul[role="listbox"] { background-color: #06101f !important; color: #e6f1ff !important; border-radius: 8px; }
    ul[role="listbox"] li { background-color: #06101f !important; color: #e6f1ff !important; }
    ul[role="listbox"] li:hover { background-color: #1e3a8a !important; }
    div[data-baseweb="select"] svg { fill: #e6f1ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# NAV
# ---------------------------------------------------------------------------
pages = ["Home", "Overview", "Travel & Performance", "Upsets & Opponent Strength", "Team Comparison"]
page = st.radio("nav", pages, horizontal=True, label_visibility="collapsed")

# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="pm-hero">
        <h1><span>⚽</span> PitchMiles</h1>
        <div class="pm-sub">Travel, rest & opponent strength vs. away-team performance · {page}</div>
        <div class="pm-accent"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# HOME
# ===========================================================================
if page == "Home":
    st.write(
        "**Research Question:** How does travel distance affect away-team performance "
        "across three geographically distinct leagues between 2014 and 2024?"
    )
    st.markdown(
        "<p style='font-size:17px; font-weight:600;'>Leagues: Premier League (England), "
        "Série A (Brazil), Major League Soccer (USA)</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches Played", "1,140")
    c2.metric("Goals Scored", "2,985")
    c3.metric("Seasons", "2014–2024")

    map_data = pd.DataFrame({
        "League": ["EPL", "MLS", "Brazilian League"],
        "Latitude": [51.509865, 37.0902, -14.2350],
        "Longitude": [-0.118092, -95.7129, -51.9253],
    })
    fig = px.scatter_mapbox(
        map_data, lat="Latitude", lon="Longitude", hover_name="League",
        color_discrete_sequence=["#64ffda"], zoom=0.5, height=400,
    )
    fig.update_layout(mapbox_style="carto-darkmatter", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    # --- Club logos ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGO_DIR = os.path.join(BASE_DIR, "..", "logos")

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
    LEAGUES = [("Premier League", premier), ("Brazilian League", brazil), ("MLS", mls)]

    st.markdown(
        "<style>.club-cell{text-align:center;margin-bottom:18px;}"
        ".club-name{display:block;margin-top:6px;font-size:14px;line-height:1.1;"
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px;"
        "margin-left:auto;margin-right:auto;}</style>",
        unsafe_allow_html=True,
    )

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
                    if os.path.exists(path):
                        st.image(path, width=logo_size)
                    st.markdown(f"<span class='club-name'>{name}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    st.header("Clubs by League")
    for league_name, club_list in LEAGUES:
        league_grid(league_name, club_list, per_row=6, logo_size=82)

# ===========================================================================
# OVERVIEW
# ===========================================================================
elif page == "Overview":
    table = st.selectbox("Select a table to view:", ["overview", "league_rankings", "home_away_pts"])
    st.dataframe(load_table(table), use_container_width=True)

    st.subheader("Team Win Percentage by League & Season")
    overview_df = load_table("overview")
    rankings_df = load_table("league_rankings")
    if not overview_df.empty and not rankings_df.empty:
        merged_df = overview_df.merge(
            rankings_df[["team", "league", "win_percentage"]], on=["team", "league"], how="left"
        )
        col_a, col_b = st.columns(2)
        selected_league = col_a.selectbox("Select League:", merged_df["league"].dropna().unique())
        selected_season = col_b.selectbox("Select Season:", merged_df["season"].dropna().unique())
        filtered = merged_df[(merged_df["league"] == selected_league) & (merged_df["season"] == selected_season)]
        if not filtered.empty:
            fig = px.pie(
                filtered, names="team", values="win_percentage", color="team",
                title=f"{selected_league}, {selected_season}", hole=0.4,
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Travel Distance by Team")
    travel_df = load_table("avg_distance_restdays")
    if not travel_df.empty:
        fig = px.bar(
            travel_df, x="avg_distance", y="team", color="league", orientation="h",
            title="Average Travel Distance by Team",
            labels={"avg_distance": "Avg Distance (km)", "team": "Team"}, height=500,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TRAVEL & PERFORMANCE
# ===========================================================================
elif page == "Travel & Performance":
    tab1, tab2 = st.tabs(["Travel Tiers", "Extreme Travel"])

    with tab1:
        table = st.selectbox("Select a table to view:", ["travel_tiers", "travel_pts_bin"], key="t1_tbl")
        st.dataframe(load_table(table), use_container_width=True)

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
        try:
            distance_away_pts = pd.read_sql(query, conn).dropna(subset=["avg_travel_km", "away_points"])
        except Exception as e:
            distance_away_pts = pd.DataFrame()
            st.warning(f"Could not build scatter: {e}")

        if distance_away_pts.empty:
            st.info("No data found for the scatter plot.")
        else:
            y_max = distance_away_pts["away_points"].max()
            fig = px.scatter(
                distance_away_pts, x="avg_travel_km", y="away_points", color="league",
                hover_name="team",
                labels={"avg_travel_km": "Avg Travel Distance (km)", "away_points": "Away Points"},
                title="Average Travel Distance vs Away Points Per Team",
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(255,255,255,0.6)")))
            fig.update_yaxes(range=[0, y_max + 20], rangemode="tozero", showgrid=True, gridwidth=0.3)
            fig.update_xaxes(showgrid=True, gridwidth=0.3)
            fig.update_layout(height=600, template="plotly_dark",
                              legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
                              margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        table = st.selectbox("Select a table to view:", ["extreme_travel", "fatigue_loss"], key="t2_tbl")
        st.dataframe(load_table(table), use_container_width=True)

        extreme_data = load_table("extreme_travel")
        if not extreme_data.empty:
            extreme_data["away_points_earned"] = pd.to_numeric(extreme_data.get("away_points_earned"), errors="coerce")
            extreme_data["days_rest"] = pd.to_numeric(extreme_data.get("days_rest"), errors="coerce")
            rest_bins = [-0.1, 50, 100, 150, 200, float("inf")]
            rest_labels = ["0–50", "51–100", "101–150", "151–200", "201+"]
            extreme_data["rest_bin"] = pd.cut(extreme_data["days_rest"], bins=rest_bins, labels=rest_labels, include_lowest=True)
            extreme_data["points_cat"] = extreme_data["away_points_earned"].astype("Int64")
            extreme_data = extreme_data[extreme_data["points_cat"].isin({0, 1, 3})].dropna(subset=["rest_bin"])

            counts = extreme_data.groupby(["rest_bin", "points_cat"]).size().reset_index(name="n")
            counts["rest_bin"] = pd.Categorical(counts["rest_bin"], categories=rest_labels, ordered=True)
            counts = counts.sort_values(["rest_bin", "points_cat"]).reset_index(drop=True)

            st.subheader("Extreme Travel — Away Points by Days of Rest")
            fig = px.bar(
                counts, x="rest_bin", y="n", color="points_cat", barmode="stack", text="n",
                labels={"rest_bin": "Days of Rest (bins)", "n": "Matches", "points_cat": "Away Points"},
                title="Distribution of Away Points by Rest Days (Extreme Travel Matches)",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=520, template="plotly_dark", legend_title_text="Away Points",
                              margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Show underlying counts"):
                st.dataframe(counts, use_container_width=True, hide_index=True)

# ===========================================================================
# UPSETS & OPPONENT STRENGTH
# ===========================================================================
elif page == "Upsets & Opponent Strength":
    tab1, tab2, tab3 = st.tabs(["ELO", "ELO vs Win %", "Upsets"])

    def compute_elo(df, k=20, base_rating=1000):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        teams = pd.concat([df["home_team"], df["away_team"]]).unique()
        ratings = {t: base_rating for t in teams}
        home_elos, away_elos = [], []
        for _, row in df.iterrows():
            R_home, R_away = ratings[row.home_team], ratings[row.away_team]
            E_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
            if row.home_score > row.away_score:
                S_home = 1
            elif row.home_score < row.away_score:
                S_home = 0
            else:
                S_home = 0.5
            ratings[row.home_team] = R_home + k * (S_home - E_home)
            ratings[row.away_team] = R_away + k * ((1 - S_home) - (1 - E_home))
            home_elos.append(ratings[row.home_team])
            away_elos.append(ratings[row.away_team])
        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        return df, ratings

    matches = pd.read_sql("SELECT league, date, home_team, away_team, home_score, away_score FROM data;", conn)
    matches_with_elo, final_ratings = compute_elo(matches)
    elo_table = (pd.DataFrame(final_ratings.items(), columns=["Team", "Elo"])
                 .sort_values("Elo", ascending=False).reset_index(drop=True))

    with tab1:
        st.subheader("What is Elo?")
        st.write("The Elo rating system numerically measures the relative skill of teams based on match results.")
        st.markdown("""
        - Every team starts at a rating of 1000.
        - After each match, points transfer between teams based on the result.
        - Beating a stronger opponent gains more points than beating a weaker one.
        - Losing to a much weaker opponent costs more than losing to an equal.
        """)
        st.subheader("How is Elo Scored?")
        st.markdown("**$R_{A_{new}} = R_{A_{old}} + K \\cdot (S_A - E_A)$**")
        st.markdown("""
        - **$R_{A_{new}}$**: new rating &nbsp; **$R_{A_{old}}$**: old rating
        - **$K$**: weight (K = 20) &nbsp; **$S_A$**: actual score (1 win / 0.5 draw / 0 loss)
        - **$E_A$**: expected score
        """)
        st.subheader("Elo Ratings Leaderboard")
        st.dataframe(elo_table, use_container_width=True)

        elo_long = pd.concat([
            matches_with_elo[["date", "league", "home_team", "home_elo"]].rename(columns={"home_team": "team", "home_elo": "elo"}),
            matches_with_elo[["date", "league", "away_team", "away_elo"]].rename(columns={"away_team": "team", "away_elo": "elo"}),
        ]).sort_values(["team", "date"])

        st.subheader("Elo Ratings Over Time")
        selected_league = st.selectbox("Select League", sorted(elo_long["league"].unique()), index=None, placeholder="Choose a league")
        if selected_league:
            teams_in_league = sorted(elo_long[elo_long["league"] == selected_league]["team"].unique())
            selected_team = st.selectbox("Select Team", teams_in_league, index=None, placeholder="Choose a team")
            if selected_team:
                filtered_elo = elo_long[(elo_long["league"] == selected_league) & (elo_long["team"] == selected_team)]
                fig = px.line(filtered_elo, x="date", y="elo",
                              title=f"Elo Rating Over Time: {selected_team} ({selected_league})",
                              labels={"elo": "Elo Rating", "date": "Date"}, template="plotly_dark")
                fig.update_traces(line=dict(width=1.5, color="#64ffda"))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        upsets = load_table("UPSETS")
        elo_local = load_table("ELO")
        if not upsets.empty and not elo_local.empty:
            home = elo_local.merge(upsets, left_on="team", right_on="home_team", how="inner")
            if {"Elo", "home_win_percentage"}.issubset(home.columns):
                st.subheader("Elo vs Home Win %")
                fig_home = px.scatter(home, x="Elo", y="home_win_percentage", color="home_team",
                                      hover_data=["home_team", "Elo", "home_win_percentage"],
                                      title="Elo vs Home Win Percentage", template="plotly_dark")
                fig_home.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_home, use_container_width=True)

            away = elo_local.merge(upsets, left_on="team", right_on="away_team", how="inner")
            if {"Elo", "away_win_percentage"}.issubset(away.columns):
                st.subheader("Elo vs Away Win %")
                fig_away = px.scatter(away, x="Elo", y="away_win_percentage", color="away_team",
                                      hover_data=["away_team", "Elo", "away_win_percentage"],
                                      title="Elo vs Away Win Percentage", template="plotly_dark")
                fig_away.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_away, use_container_width=True)
        else:
            st.info("Elo / Upsets tables unavailable.")

    with tab3:
        upsets = load_table("UPSETS")
        if not upsets.empty and "season" in upsets.columns:
            season_counts = upsets.groupby("season").size().reset_index(name="Upset_Count").sort_values("season")
            fig_upsets = px.line(season_counts, x="season", y="Upset_Count", markers=True,
                                 title="Upsets per Season", template="plotly_dark")
            fig_upsets.update_traces(line=dict(color="#64ffda"))
            fig_upsets.update_layout(xaxis_title="Season", yaxis_title="Number of Upsets")
            st.plotly_chart(fig_upsets, use_container_width=True)

            st.subheader("Upset Matches")
            display = upsets.rename(columns={
                "league": "League", "season": "Season", "home_team": "Home Team",
                "away_team": "Away Team", "home_score": "Home Score",
                "home_win_percentage": "Home Win %", "away_win_percentage": "Away Win %",
            })
            cols = [c for c in ["League", "Season", "Home Team", "Away Team", "Home Score", "Home Win %", "Away Win %"] if c in display.columns]
            st.dataframe(display[cols], use_container_width=True)

# ===========================================================================
# TEAM COMPARISON
# ===========================================================================
else:
    st.write("Compare three teams across leagues.")
    rankings = load_table("league_rankings")
    travel = load_table("avg_distance_restdays")
    home_away = load_table("home_away_pts")

    if rankings.empty or travel.empty or home_away.empty:
        st.info("Comparison data unavailable.")
    else:
        comparison = (rankings
                      .merge(travel, on=["league", "team"], how="left")
                      .merge(home_away[["league", "team", "home_points", "away_points", "total_points"]],
                             on=["league", "team"], how="left"))

        options = [f"{row['team']} ({row['league']})" for _, row in comparison.iterrows()]
        c1, c2, c3 = st.columns(3)
        team1 = c1.selectbox("Team 1", options, index=0)
        team2 = c2.selectbox("Team 2", options, index=min(1, len(options) - 1))
        team3 = c3.selectbox("Team 3", options, index=min(2, len(options) - 1))

        def row_for(label):
            name = label.split(" (")[0]
            return comparison[comparison["team"] == name].iloc[0]

        if len({team1, team2, team3}) < 3:
            st.error("Please select three different teams to compare.")
        else:
            for col, label in zip(st.columns(3), [team1, team2, team3]):
                d = row_for(label)
                with col:
                    st.markdown(f"**{label}**")
                    st.metric("Win %", f"{d.get('win_percentage', float('nan')):.1f}%")
                    st.metric("Avg Travel", f"{d.get('avg_distance', float('nan')):.1f} km")
                    st.metric("Avg Rest Days", f"{d.get('avg_restdays', float('nan')):.1f} days")
                    st.metric("Away Points", f"{d.get('away_points', float('nan')):.1f}")