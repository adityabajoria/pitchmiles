import os
import sqlite3
from pathlib import Path
from itertools import islice

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
    return sqlite3.connect(DB_PATH, check_same_thread=False)


conn = get_conn()


def load_table(name):
    try:
        return pd.read_sql(f'SELECT * FROM "{name}"', conn)
    except Exception as e:
        st.warning(f"Could not load table '{name}': {e}")
        return pd.DataFrame()


def note(text):
    """Render a research-note callout box."""
    st.markdown(f"<div class='pm-note'>{text}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0a192f !important; color: #e6f1ff !important; }
    [data-testid="stSidebar"] { background-color: #0a192f !important; }
    html, body, .stApp, .stMarkdown, [class*="st-"],
    [data-testid="stMarkdownContainer"], label { color: #e6f1ff !important; }
    #MainMenu, footer, header {visibility: hidden;}

    .pm-hero {
        background: linear-gradient(135deg, #0a192f 0%, #112d4e 45%, #1e3a8a 100%);
        border: 1px solid #1e3a8a; border-radius: 16px; padding: 26px 32px;
        margin: 4px 0 12px 0; box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .pm-hero h1 { margin:0; font-size:40px; font-weight:800; letter-spacing:.5px;
        color:#fff !important; display:flex; align-items:center; gap:14px; }
    .pm-hero .pm-sub { margin-top:6px; font-size:15px; font-weight:500; color:#9ecbff !important; }
    .pm-accent { height:4px; width:120px; margin-top:14px; border-radius:2px;
        background: linear-gradient(90deg,#3A9BDC,#64ffda); }

    /* Research note callouts */
    .pm-note {
        background:#0d2140; border-left:4px solid #64ffda; border-radius:8px;
        padding:14px 18px; margin:10px 0 18px 0; font-size:15px; line-height:1.6;
        color:#cfe3ff !important;
    }
    .pm-section {
        font-size:13px; letter-spacing:2px; text-transform:uppercase;
        color:#64ffda !important; font-weight:700; margin-top:8px;
    }

    /* Radio nav one line */
    div[role="radiogroup"] {
        gap:8px; flex-wrap:nowrap; overflow-x:auto; white-space:nowrap;
        background:#0d2140; padding:8px 10px; border-radius:12px; border:1px solid #16345c;
    }
    div[role="radiogroup"] label { padding:8px 16px; border-radius:8px; cursor:pointer;
        transition:background .15s ease; white-space:nowrap; flex-shrink:0; }
    div[role="radiogroup"] label:hover { background:#16345c; }

    [data-testid="stMetric"] { background:#0d2140; border:1px solid #16345c;
        border-radius:12px; padding:14px 16px; }
    thead tr th { background-color:#0d2140 !important; color:#e6f1ff !important; }
    tbody tr td { background-color:#fff !important; color:#0a192f !important; }
    div[data-baseweb="select"] > div { background-color:#0d2140 !important; color:#e6f1ff !important;
        border-radius:8px; border:1px solid #1e3a8a !important; }
    ul[role="listbox"] { background-color:#06101f !important; }
    ul[role="listbox"] li { background-color:#06101f !important; color:#e6f1ff !important; }
    ul[role="listbox"] li:hover { background-color:#1e3a8a !important; }
    div[data-baseweb="select"] svg { fill:#e6f1ff !important; }

    /* Logo grid: native columns, centered captions, no fighting flexbox */
    div[data-testid="stImage"] { display:flex; justify-content:center; }
    .club-name { display:block; text-align:center; margin-top:6px; font-size:13px;
        font-weight:600; color:#cfe3ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = ["Home", "Overview", "Travel & Performance", "Upsets & Opponent Strength", "Team Comparison & Analysis"]
page = st.radio("nav", pages, horizontal=True, label_visibility="collapsed")

st.markdown(
    f"""
    <div class="pm-hero">
        <h1><span>⚽</span> PitchMiles</h1>
        <div class="pm-sub">Measuring how away travel affects points per game, after adjusting for opponent strength and rest · {page}</div>
        <div class="pm-accent"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

# ===========================================================================
# HOME
# ===========================================================================
if page == "Home":
    st.markdown(
        "<div class='pm-note' style='border-left:4px solid #64ffda;'>"
        "<b>Research question:</b> From 2014 to 2024, how much does each 100 km of away travel change a team's "
        "points per game across England, Brazil, and MLS, after adjusting for opponent strength and rest?"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='pm-section'>Abstract</div>", unsafe_allow_html=True)
    st.markdown(
        "Away teams win less often than home teams, a pattern so consistent it is treated as a rule of the sport. This "
        "study isolates one possible cause, travel fatigue, and measures how much the distance a team travels and the rest "
        "it gets affect the points it earns on the road. It draws on ten seasons of match data from 2014 to 2024 across "
        "England's Premier League, Brazil's Série A, and Major League Soccer, three competitions chosen because their very "
        "different geographies produce very different travel demands. The central test is whether any travel effect "
        "survives once the largest driver of results, the strength of the opponent, is properly accounted for."
    )

    note(
        "<b>Why three leagues?</b> Travel distance only becomes a meaningful variable when geography makes it one. England's "
        "clubs sit close together, so away trips are short and vary little from week to week, which leaves almost no range "
        "to study. Brazil and the United States span thousands of kilometres, producing the wide spread of travel distances "
        "needed to detect a fatigue effect if one exists. Comparing all three leagues also reveals whether any effect scales "
        "with distance, which is exactly what the research question asks."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches Analysed", "1,140")
    c2.metric("Goals Scored", "2,985")
    c3.metric("Seasons", "2014–2024")

    st.markdown("<div class='pm-section'>The Three Leagues</div>", unsafe_allow_html=True)
    st.markdown(
        "The map places the three competitions in their geographic context. The wide spatial spread is the point of the "
        "design, since it supplies the range of travel distances that the research question depends on. Fixtures range "
        "from short local derbies in England to cross country journeys in Brazil and the United States, and that contrast "
        "is what makes a travel effect measurable."
    )
    map_data = pd.DataFrame({
        "League": ["EPL", "MLS", "Brazilian League"],
        "Latitude": [51.509865, 37.0902, -14.2350],
        "Longitude": [-0.118092, -95.7129, -51.9253],
    })
    fig = px.scatter_mapbox(map_data, lat="Latitude", lon="Longitude", hover_name="League",
                            color_discrete_sequence=["#64ffda"], zoom=0.5, height=380)
    fig.update_layout(mapbox_style="carto-darkmatter", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    # --- Structured logo grid ---
    st.markdown("<div class='pm-section'>Clubs by League</div>", unsafe_allow_html=True)
    st.markdown("The clubs analysed from each league, chosen as consistent top sides across the study period. Holding the sample to "
        "comparable teams keeps overall quality broadly similar within each competition, which helps separate the effect of "
        "travel from the effect of simply being a stronger or weaker side.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGO_DIR = os.path.join(BASE_DIR, "..", "logos")
    premier = [
        ("Man City", "prem/Manchester_City_FC_badge.svg"), ("Liverpool", "prem/Liverpool_FC.svg"),
        ("Arsenal", "prem/Arsenal_FC.svg"), ("Chelsea", "prem/Chelsea_FC.svg"),
        ("Man United", "prem/Manchester_United_FC_crest.svg"), ("Tottenham", "prem/Tottenham_Hotspur.svg"),
        ("Newcastle", "prem/Newcastle_United_Logo.svg"), ("Leicester City", "prem/Leicester_City_crest.svg"),
        ("Aston Villa", "prem/Aston_Villa_FC_new_crest.svg"), ("West Ham", "prem/West_Ham_United_FC_logo.svg"),
    ]
    brazil = [
        ("Sao Paulo FC", "brazilian/Brasao_do_Sao_Paulo_Futebol_Clube.svg"),
        ("Atletico Mineiro", "brazilian/Clube_Atlético_Mineiro_crest.svg"),
        ("Flamengo", "brazilian/Flamengo_braz_logo.svg"), ("Palmeiras", "brazilian/Palmeiras_logo.svg"),
        ("Corinthians", "brazilian/Sport_Club_Corinthians_Paulista_Logo.png"),
        ("Internacional", "brazilian/SC_Internacional_Brazil_Logo.svg"),
        ("Santos FC", "brazilian/Santos_Logo.png"), ("Gremio", "brazilian/Gremio_logo.svg"),
        ("Botafogo", "brazilian/Botafogo_de_Futebol_e_Regatas_logo.svg"),
        ("Cruzeiro", "brazilian/Cruzeiro_Esporte_Clube_(logo).svg"),
    ]
    mls = [
        ("LA Galaxy", "mls/Los_Angeles_Galaxy_logo.svg"), ("DC United", "mls/D.C._United_logo_(2016).svg"),
        ("Houston Dynamo", "mls/Houston_Dynamo_FC_logo.svg"), ("Seattle Sounders", "mls/Seattle_Sounders_logo.svg"),
        ("Sporting KC", "mls/Sporting_Kansas_City_logo.svg"), ("Chicago Fire", "mls/Chicago_Fire_logo,_2021.svg"),
        ("LAFC", "mls/Los_Angeles_Football_Club.svg"), ("NY Red Bulls", "mls/New_York_Red_Bulls_logo.svg"),
        ("Portland Timbers", "mls/Portland_Timbers_logo.svg"), ("Philadelphia Union", "mls/Philadelphia_Union_2018_logo.svg"),
    ]

    def logo_grid(title, clubs, per_row=5):
        st.markdown(f"#### {title}")
        for start in range(0, len(clubs), per_row):
            row = clubs[start:start + per_row]
            cols = st.columns(per_row)
            for i in range(per_row):
                with cols[i]:
                    if i < len(row):
                        name, rel = row[i]
                        path = os.path.join(LOGO_DIR, rel)
                        if os.path.exists(path):
                            st.image(path, width=64)
                        else:
                            st.markdown(
                                "<div style='text-align:center;font-size:40px'>⚽</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(f"<span class='club-name'>{name}</span>", unsafe_allow_html=True)

    logo_grid("Premier League", premier)
    logo_grid("Brazilian League", brazil)
    logo_grid("Major League Soccer", mls)

# ===========================================================================
# OVERVIEW
# ===========================================================================
elif page == "Overview":
    st.markdown("<div class='pm-section'>Descriptive Landscape</div>", unsafe_allow_html=True)
    st.markdown(
        "This page establishes the basic shape of the data before any effect is tested: who wins, who travels, and how those "
        "quantities are distributed across the three leagues. These views are descriptive context rather than evidence, and "
        "they deliberately make no causal claim. They set the baseline picture that any later statement about travel must "
        "be checked against."
    )

    table = st.selectbox("Inspect a summary table:", ["overview", "league_rankings", "home_away_pts"])
    st.dataframe(load_table(table), use_container_width=True)

    st.markdown("<div class='pm-section'>Competitive Balance</div>", unsafe_allow_html=True)
    st.markdown(
        "This chart shows how win percentage is shared among teams within a chosen league and season. It matters for the "
        "research question because the degree of concentration shapes how results should be read: in a top heavy league, a "
        "few strong sides win most matches whatever the travel involved. That concentration can either hide a real travel "
        "effect or imitate one, which is precisely why opponent strength has to be controlled before travel is judged."
    )
    overview_df, rankings_df = load_table("overview"), load_table("league_rankings")
    if not overview_df.empty and not rankings_df.empty:
        merged = overview_df.merge(rankings_df[["team", "league", "win_percentage"]], on=["team", "league"], how="left")
        ca, cb = st.columns(2)
        lg = ca.selectbox("League:", merged["league"].dropna().unique())
        ssn = cb.selectbox("Season:", merged["season"].dropna().unique())
        f = merged[(merged["league"] == lg) & (merged["season"] == ssn)]
        if not f.empty:
            fig = px.pie(f, names="team", values="win_percentage", color="team",
                         title=f"Win % Distribution: {lg}, {ssn}", hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='pm-section'>Travel Burden by Team</div>", unsafe_allow_html=True)
    st.markdown(
        "This chart shows the average travel distance per away fixture for each team, the core exposure variable in the study. "
        "The differences in scale across leagues are large, with Brazilian and American sides routinely covering distances "
        "that English clubs never face. That contrast previews where a fatigue effect should be easiest to detect, since a "
        "penalty tied to distance should surface first among the teams that travel farthest."
    )
    travel_df = load_table("avg_distance_restdays")
    if not travel_df.empty:
        fig = px.bar(travel_df, x="avg_distance", y="team", color="league", orientation="h",
                     title="Average Travel Distance by Team",
                     labels={"avg_distance": "Avg Distance (km)", "team": "Team"}, height=520)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TRAVEL & PERFORMANCE
# ===========================================================================
elif page == "Travel & Performance":
    st.markdown("<div class='pm-section'>The Core Question</div>", unsafe_allow_html=True)
    st.markdown(
        "This page confronts the research question directly: does travelling further, or resting less, actually cost teams "
        "points on the road? It approaches the question in two steps, first looking at the broad relationship between travel "
        "and away points across all fixtures, then narrowing to the most extreme journeys where any effect should be "
        "largest. Reading the two views together shows whether a travel penalty is present in general, or only in the "
        "hardest cases, or not clearly at all."
    )
    tab1, tab2 = st.tabs(["Travel Tiers", "Extreme Travel"])

    with tab1:
        note(
            "<b>How to read this:</b> each point is a team, plotting its average away travel distance against its average away "
            "points. If travel fatigue were a strong force, the points would trend downward, with greater distance linked to "
            "fewer points earned. A flat or scattered cloud instead indicates that travel is at most a minor factor next to "
            "team quality, which is the pattern the rest of the analysis examines in more detail."
        )
        table = st.selectbox("Inspect a table:", ["travel_tiers", "travel_pts_bin"], key="t1_tbl")
        st.dataframe(load_table(table), use_container_width=True)

        query = """
        SELECT h.league, h.team, CAST(h.away_points AS FLOAT) AS away_points, d.avg_travel_km
        FROM home_away_pts AS h
        JOIN (SELECT league, away_team AS team, AVG(distance_km) AS avg_travel_km
              FROM data WHERE distance_km IS NOT NULL GROUP BY league, away_team) AS d
        ON h.league = d.league AND h.team = d.team
        WHERE h.away_points IS NOT NULL
        """
        try:
            dap = pd.read_sql(query, conn).dropna(subset=["avg_travel_km", "away_points"])
        except Exception as e:
            dap = pd.DataFrame(); st.warning(f"Could not build scatter: {e}")

        if dap.empty:
            st.info("No data found for the scatter plot.")
        else:
            ymax = dap["away_points"].max()
            fig = px.scatter(dap, x="avg_travel_km", y="away_points", color="league", hover_name="team",
                             labels={"avg_travel_km": "Avg Travel Distance (km)", "away_points": "Away Points"},
                             title="Average Travel Distance vs Away Points")
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(255,255,255,0.6)")))
            fig.update_yaxes(range=[0, ymax + 20]); fig.update_layout(height=560, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            note(
                "<b>What we tend to see:</b> the relationship is weak, and the three leagues overlap heavily rather than separating "
                "by distance. Strong teams tend to score well away regardless of how far they travel, while weaker teams "
                "struggle even on short trips. This is the first indication that travel distance on its own is a poor "
                "predictor of away points, a reading the formal out of sample model later confirms."
            )

    with tab2:
        note(
            "<b>Why extremes matter:</b> a genuine but small effect can be invisible in ordinary fixtures and only emerge in the "
            "hardest cases, the longest trips taken on the least rest. This view isolates the most extreme travel matches and "
            "groups them by days of rest, then shows how away results are distributed within each group. If fatigue drives "
            "results anywhere, this is where it should show up most clearly."
        )
        table = st.selectbox("Inspect a table:", ["extreme_travel", "fatigue_loss"], key="t2_tbl")
        st.dataframe(load_table(table), use_container_width=True)

        ext = load_table("extreme_travel")
        if not ext.empty:
            ext["away_points_earned"] = pd.to_numeric(ext.get("away_points_earned"), errors="coerce")
            ext["days_rest"] = pd.to_numeric(ext.get("days_rest"), errors="coerce")
            bins = [-0.1, 50, 100, 150, 200, float("inf")]
            labels = ["0–50", "51–100", "101–150", "151–200", "201+"]
            ext["rest_bin"] = pd.cut(ext["days_rest"], bins=bins, labels=labels, include_lowest=True)
            ext["points_cat"] = ext["away_points_earned"].astype("Int64")
            ext = ext[ext["points_cat"].isin({0, 1, 3})].dropna(subset=["rest_bin"])
            counts = ext.groupby(["rest_bin", "points_cat"]).size().reset_index(name="n")
            counts["rest_bin"] = pd.Categorical(counts["rest_bin"], categories=labels, ordered=True)
            counts = counts.sort_values(["rest_bin", "points_cat"])

            fig = px.bar(counts, x="rest_bin", y="n", color="points_cat", barmode="stack", text="n",
                         labels={"rest_bin": "Days of Rest", "n": "Matches", "points_cat": "Away Points"},
                         title="Away Outcomes by Rest (Extreme-Travel Matches)")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=520, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            note(
                "<b>Interpretation:</b> even on these most demanding journeys, wins and draws remain common rather than rare. The "
                "absence of a sharp collapse in away results under extreme travel is itself an informative finding. It places "
                "an upper bound on how large a fatigue penalty can plausibly be, since the effect stays modest even in the "
                "conditions designed to expose it."
            )

# ===========================================================================
# UPSETS & OPPONENT STRENGTH
# ===========================================================================
elif page == "Upsets & Opponent Strength":
    st.markdown("<div class='pm-section'>The Confounder</div>", unsafe_allow_html=True)
    st.markdown(
        "Opponent strength is the single largest driver of match results, so it has to be measured before any travel effect "
        "can be trusted. This page builds an Elo rating that summarises each team's strength, then shows why leaving it "
        "uncontrolled would distort the estimate. A strong team can win after a long trip and a weak team can lose after a "
        "short one, so without adjusting for the opponent, travel can easily be blamed or excused for results it did not cause."
    )
    tab1, tab2, tab3 = st.tabs(["Elo Explained", "Elo vs Win %", "Upsets"])

    def compute_elo(df, k=20, base=1000):
        df = df.copy(); df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
        teams = pd.concat([df["home_team"], df["away_team"]]).unique()
        r = {t: base for t in teams}; he, ae = [], []
        for _, row in df.iterrows():
            rh, ra = r[row.home_team], r[row.away_team]
            eh = 1 / (1 + 10 ** ((ra - rh) / 400))
            sh = 1 if row.home_score > row.away_score else (0 if row.home_score < row.away_score else 0.5)
            r[row.home_team] = rh + k * (sh - eh); r[row.away_team] = ra + k * ((1 - sh) - (1 - eh))
            he.append(r[row.home_team]); ae.append(r[row.away_team])
        df["home_elo"], df["away_elo"] = he, ae
        return df, r

    matches = pd.read_sql("SELECT league, date, home_team, away_team, home_score, away_score FROM data;", conn)
    mwe, final_ratings = compute_elo(matches)
    elo_leaderboard = (pd.DataFrame(final_ratings.items(), columns=["Team", "Elo"])
                       .sort_values("Elo", ascending=False).reset_index(drop=True))

    with tab1:
        st.markdown(
            "Elo converts a team's full history of results into a single strength number that updates after every match. Each "
            "team starts from the same rating, and points transfer from the loser to the winner in proportion to how "
            "surprising the result was, so beating a strong side is worth more than beating a weak one. The result is a "
            "continuous, always current measure of quality that can be held constant when testing whether travel matters on "
            "its own."
        )
        st.markdown("Elo update: $R_{new} = R_{old} + K (S - E)$, where $K$ controls how much each match moves the rating, $S$ is the "
                    "actual result (1 for a win, 0.5 for a draw, 0 for a loss), and $E$ is the result expected from the rating "
                    "gap before kickoff.")
        st.markdown("<div class='pm-section'>Strength Leaderboard</div>", unsafe_allow_html=True)
        st.dataframe(elo_leaderboard, use_container_width=True)

        elo_long = pd.concat([
            mwe[["date", "league", "home_team", "home_elo"]].rename(columns={"home_team": "team", "home_elo": "elo"}),
            mwe[["date", "league", "away_team", "away_elo"]].rename(columns={"away_team": "team", "away_elo": "elo"}),
        ]).sort_values(["team", "date"])
        st.markdown("<div class='pm-section'>Elo Trajectories</div>", unsafe_allow_html=True)
        st.markdown("Tracing a team's Elo over time reveals its form arc, including rises, sustained peaks, and declines that a single "
                    "season average would smooth away. This context matters for the research question, since the same club can "
                    "absorb long travel comfortably in a strong season and struggle with it in a weak one. Reading travel "
                    "effects without accounting for form would therefore be misleading.")
        lg = st.selectbox("League", sorted(elo_long["league"].unique()), index=None, placeholder="Choose a league")
        if lg:
            tms = sorted(elo_long[elo_long["league"] == lg]["team"].unique())
            tm = st.selectbox("Team", tms, index=None, placeholder="Choose a team")
            if tm:
                fe = elo_long[(elo_long["league"] == lg) & (elo_long["team"] == tm)]
                fig = px.line(fe, x="date", y="elo", title=f"Elo Over Time: {tm} ({lg})",
                              labels={"elo": "Elo Rating", "date": "Date"})
                fig.update_traces(line=dict(width=1.5, color="#64ffda")); fig.update_layout(height=480, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        note(
            "<b>The key relationship:</b> this view checks whether Elo tracks actual success, which it should if it is a good "
            "measure of strength. A tight link between Elo and win percentage confirms that opponent quality explains most of "
            "the variation in results. That is exactly why strength must be removed from the picture before any remaining "
            "variation can fairly be attributed to travel."
        )
        elo_tbl = load_table("ELO")          # columns: index, Team, Elo
        upsets = load_table("UPSETS")
        if not elo_tbl.empty and not upsets.empty:
            # FIX: ELO table's column is 'Team' (capital), join to home/away team names
            home = elo_tbl.merge(upsets, left_on="Team", right_on="home_team", how="inner")
            if {"Elo", "home_win_percentage"}.issubset(home.columns):
                st.markdown("<div class='pm-section'>Elo vs Home Win %</div>", unsafe_allow_html=True)
                fh = px.scatter(home, x="Elo", y="home_win_percentage", color="home_team",
                                hover_data=["home_team", "Elo", "home_win_percentage"],
                                title="Elo vs Home Win Percentage")
                fh.update_layout(height=560, showlegend=False, **PLOTLY_LAYOUT)
                st.plotly_chart(fh, use_container_width=True)

            away = elo_tbl.merge(upsets, left_on="Team", right_on="away_team", how="inner")
            if {"Elo", "away_win_percentage"}.issubset(away.columns):
                st.markdown("<div class='pm-section'>Elo vs Away Win %</div>", unsafe_allow_html=True)
                fa = px.scatter(away, x="Elo", y="away_win_percentage", color="away_team",
                                hover_data=["away_team", "Elo", "away_win_percentage"],
                                title="Elo vs Away Win Percentage")
                fa.update_layout(height=560, showlegend=False, **PLOTLY_LAYOUT)
                st.plotly_chart(fa, use_container_width=True)
            note(
                "<b>Takeaway:</b> the clear upward trend in both panels confirms that Elo captures most of what determines results. "
                "This is the confounder the formal model controls for, and it is a strong one. Once opponent strength is held "
                "constant, the apparent effect of travel shrinks sharply, which is the central result the study reports "
                "honestly rather than overstating."
            )
        else:
            st.info("Elo / Upsets tables unavailable.")

    with tab3:
        st.markdown(
            "An upset, where a weaker team beats a stronger one, is exactly the kind of result where a small effect like travel "
            "fatigue could plausibly tip the balance. Tracking how often upsets occur across seasons tests whether results "
            "are becoming more or less predictable over time. A stable upset rate suggests the underlying balance between "
            "strength and chance has not shifted much across the study period."
        )
        upsets = load_table("UPSETS")
        if not upsets.empty and "season" in upsets.columns:
            sc = upsets.groupby("season").size().reset_index(name="Upsets").sort_values("season")
            fig = px.line(sc, x="season", y="Upsets", markers=True, title="Upsets per Season")
            fig.update_traces(line=dict(color="#64ffda")); fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            disp = upsets.rename(columns={"league": "League", "season": "Season", "home_team": "Home",
                                          "away_team": "Away", "home_score": "Home Score",
                                          "home_win_percentage": "Home Win %", "away_win_percentage": "Away Win %"})
            cols = [c for c in ["League", "Season", "Home", "Away", "Home Score", "Home Win %", "Away Win %"] if c in disp.columns]
            st.dataframe(disp[cols], use_container_width=True)

# ===========================================================================
# TEAM COMPARISON
# ===========================================================================
else:
    st.markdown("<div class='pm-section'>Side-by-Side Profiles</div>", unsafe_allow_html=True)
    st.markdown(
        "Comparing individual clubs turns the study's abstract variables into something concrete. Selecting three teams, "
        "ideally from different leagues, shows how their travel burden, rest, and away returns line up side by side. A club "
        "like Seattle travels far more than a London side, yet its away points stay broadly comparable, which puts the "
        "study's central tension in a single view: distance differs enormously while results do not follow it closely."
    )
    rankings, travel, home_away = load_table("league_rankings"), load_table("avg_distance_restdays"), load_table("home_away_pts")

    if rankings.empty or travel.empty or home_away.empty:
        st.info("Comparison data unavailable.")
    else:
        comp = (rankings.merge(travel, on=["league", "team"], how="left")
                .merge(home_away[["league", "team", "home_points", "away_points", "total_points"]],
                       on=["league", "team"], how="left"))
        options = [f"{r['team']} ({r['league']})" for _, r in comp.iterrows()]
        c1, c2, c3 = st.columns(3)
        t1 = c1.selectbox("Team 1", options, index=0)
        t2 = c2.selectbox("Team 2", options, index=min(1, len(options) - 1))
        t3 = c3.selectbox("Team 3", options, index=min(2, len(options) - 1))

        def row_for(label): return comp[comp["team"] == label.split(" (")[0]].iloc[0]

        if len({t1, t2, t3}) < 3:
            st.error("Please select three different teams to compare.")
        else:
            rows = [row_for(t1), row_for(t2), row_for(t3)]
            names = [t1, t2, t3]
            metrics = {"Win %": "win_percentage", "Avg Travel (km)": "avg_distance",
                       "Avg Rest (days)": "avg_restdays", "Away Points": "away_points"}

            # --- Summary table first ---
            st.markdown("<div class='pm-section'>Summary Table</div>", unsafe_allow_html=True)
            st.markdown(
                "This table lists the raw figures for each selected club, so the comparison rests on numbers rather than impressions. "
                "Reading across a row compares the three teams on one dimension at a time. The largest gap is almost always in "
                "travel distance, while away points remain comparatively close, which reflects the study's finding that "
                "distance and results are only loosely connected."
            )
            tbl = pd.DataFrame({
                "Metric": list(metrics.keys()),
                **{nm: [round(float(r.get(c, np.nan)), 1) for c in metrics.values()] for nm, r in zip(names, rows)}
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            # --- Radar chart across normalized metrics ---
            st.markdown("<div class='pm-section'>Multi-Metric Profile</div>", unsafe_allow_html=True)
            note(
                "The radar plot rescales every metric to a shared 0 to 1 range, with each team measured relative to the highest "
                "value among the three, so quantities on very different units can be compared on one shape. A team that leads "
                "a given metric reaches the outer edge on that axis. The overall silhouette then gives an at a glance profile "
                "of each club's strengths and burdens, making differences in travel and quality easy to see together."
            )
            import plotly.graph_objects as go
            radar_metrics = list(metrics.keys())
            # normalise each metric across the three teams (max = 1)
            raw = {nm: [float(r.get(c, np.nan)) for c in metrics.values()] for nm, r in zip(names, rows)}
            maxes = [max(abs(raw[nm][i]) for nm in names) or 1 for i in range(len(radar_metrics))]
            fig_r = go.Figure()
            palette = ["#64ffda", "#3A9BDC", "#9ecbff"]
            for nm, color in zip(names, palette):
                vals = [raw[nm][i] / maxes[i] for i in range(len(radar_metrics))]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=radar_metrics + [radar_metrics[0]],
                    fill="toself", name=nm, line=dict(color=color),
                ))
            fig_r.update_layout(
                height=520, polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#16345c"),
                                    angularaxis=dict(gridcolor="#16345c"), bgcolor="rgba(0,0,0,0)"),
                **PLOTLY_LAYOUT)
            st.plotly_chart(fig_r, use_container_width=True)

    # --- Findings section tying in the signal study ---
    st.markdown("---")
    st.markdown("<div class='pm-section'>Final Analysis</div>", unsafe_allow_html=True)
    note(
        "The exploration moved deliberately from description to formal testing. The descriptive views set the stage: "
        "away teams consistently earn fewer points than home teams, and the three leagues differ enormously in how far "
        "their clubs travel, with Brazilian and American sides covering distances English clubs never face. Yet the core "
        "scatter of average travel against away points showed only a weak, heavily overlapped relationship, and even the "
        "most extreme journeys on the shortest rest still produced a healthy share of wins and draws. Those patterns "
        "already suggested that raw distance is a poor guide to away results, and that something else was doing most of "
        "the explaining. Building an Elo rating identified what that something was: opponent strength tracks win "
        "percentage closely, marking it as the dominant driver of outcomes and the key variable any travel estimate has "
        "to be adjusted for."
    )
    note(
        "The formal experiments confirmed this reading. On the Premier League, using a temporal train and test split and "
        "controlling for pre match Elo, travel distance is statistically significant in sample, at about −0.05 away "
        "points per standard deviation of distance with p = 0.011. Once the model is asked to predict unseen matches, "
        "however, travel adds essentially no value beyond what opponent strength already explains. The key finding is "
        "therefore precise rather than flashy: travel fatigue is a real but very small effect that is almost entirely "
        "absorbed by opponent quality, so it is not independently useful for predicting away results across these three "
        "leagues. Equally important is the methodological lesson, that a signal can be genuine in sample and still fail "
        "out of sample, which is why the study leans on out of sample validation and reports a modest, honest result "
        "rather than overstating a weak one."
    )