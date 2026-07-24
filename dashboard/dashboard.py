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

pages = ["Home", "Overview", "Travel & Performance", "Upsets & Opponent Strength", "Team Comparison"]
page = st.radio("nav", pages, horizontal=True, label_visibility="collapsed")

st.markdown(
    f"""
    <div class="pm-hero">
        <h1><span>⚽</span> PitchMiles</h1>
        <div class="pm-sub">Does travel fatigue move the needle on away-team results? · {page}</div>
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
    st.markdown("<div class='pm-section'>Abstract</div>", unsafe_allow_html=True)
    st.markdown(
        "Away teams win less often than home teams — a pattern so consistent it is treated as a law of the sport. "
        "**PitchMiles** asks how much of that away disadvantage can be attributed to *travel fatigue* specifically: "
        "the burden of long journeys and short rest between fixtures. Using ten seasons (2014–2024) of match data from "
        "three geographically distinct leagues — the **Premier League** (compact England), **Série A** (continental Brazil), "
        "and **MLS** (continental USA) — we examine whether the distance a team travels and the rest it gets predict how "
        "many points it takes on the road, and critically, whether any such effect survives once we account for the "
        "single largest driver of results: the strength of the opponent."
    )

    note(
        "<b>Why three leagues?</b> Travel distance only varies meaningfully when geography does. England's clubs are "
        "packed into a small area, so travel barely differs between fixtures. Brazil and the USA span thousands of "
        "kilometres, creating the natural variation needed to isolate a fatigue signal. Comparing across the three lets "
        "us see whether any travel effect scales with the actual distances involved."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches Analysed", "1,140")
    c2.metric("Goals Scored", "2,985")
    c3.metric("Seasons", "2014–2024")

    st.markdown("<div class='pm-section'>The Three Leagues</div>", unsafe_allow_html=True)
    st.markdown(
        "The map below anchors the three competitions geographically. The spatial spread is the whole point: it is what "
        "gives the study its range of travel distances, from short intra-city derbies to cross-continental trips."
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
    st.markdown("A representative sample of clubs from each competition included in the dataset.")

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
        "Before testing any hypothesis, we establish the lay of the land: who wins, who travels, and how those quantities "
        "are distributed across the three leagues. These descriptive views are not yet evidence of a travel effect — they "
        "are the baseline picture any causal claim must be checked against."
    )

    table = st.selectbox("Inspect a summary table:", ["overview", "league_rankings", "home_away_pts"])
    st.dataframe(load_table(table), use_container_width=True)

    st.markdown("<div class='pm-section'>Competitive Balance</div>", unsafe_allow_html=True)
    st.markdown(
        "The chart below shows how win percentage is distributed among teams within a chosen league and season. It matters "
        "for the fatigue question because it reveals how *concentrated* success is: in a top-heavy league, a handful of "
        "strong sides win most matches regardless of travel, which can mask or mimic a fatigue signal if not controlled for."
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
                         title=f"Win % Distribution — {lg}, {ssn}", hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='pm-section'>Travel Burden by Team</div>", unsafe_allow_html=True)
    st.markdown(
        "Average travel distance per away fixture, by team. This is the raw exposure variable at the heart of the study. "
        "Note the scale differences across leagues — Brazilian and American sides routinely cover distances that would be "
        "unthinkable in England — which previews why a fatigue effect, if it exists anywhere, should be most visible there."
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
        "This is the heart of the analysis: does travelling further, or resting less, actually cost teams points on the road? "
        "We approach it two ways — first by grouping fixtures into travel tiers, then by isolating the most extreme journeys "
        "where any fatigue effect should be largest."
    )
    tab1, tab2 = st.tabs(["Travel Tiers", "Extreme Travel"])

    with tab1:
        note(
            "<b>How to read this:</b> each point is a team, plotting its average away travel against the points it earns away "
            "from home. If travel fatigue were a dominant force, we would expect a clear downward slope — more kilometres, "
            "fewer points. A flat or noisy cloud instead suggests travel is, at most, a minor factor relative to team quality."
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
                "<b>What we tend to see:</b> the relationship is weak and heavily overlapped across leagues. Strong teams "
                "score well away regardless of distance; weak teams struggle even on short trips. This is the first hint that "
                "travel alone is a poor predictor — a hint the formal signal study later confirms."
            )

    with tab2:
        note(
            "<b>Why extremes matter:</b> if a fatigue effect is real but small, it will be drowned out in typical fixtures and "
            "only surface in the hardest cases — the longest trips on the shortest rest. Here we bucket the most extreme-travel "
            "matches by days of rest and look at the distribution of away outcomes."
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
                "<b>Interpretation:</b> even in these worst-case journeys, wins (3 points) and draws (1) remain common. The "
                "absence of a stark collapse in results under extreme travel is itself a finding — it bounds how large any "
                "fatigue penalty can plausibly be."
            )

# ===========================================================================
# UPSETS & OPPONENT STRENGTH
# ===========================================================================
elif page == "Upsets & Opponent Strength":
    st.markdown("<div class='pm-section'>The Confounder</div>", unsafe_allow_html=True)
    st.markdown(
        "Any honest attempt to measure travel fatigue must reckon with the dominant driver of match outcomes: **how good the "
        "opponent is**. We quantify team strength with an Elo rating, then show why controlling for it is essential — a strong "
        "team on a long trip may still win, and mistaking that for 'travel doesn't matter' (or the reverse) is the central "
        "trap this section guards against."
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
            "**Elo** converts a history of results into a single strength number. Every team starts at 1000; after each match "
            "points transfer from loser to winner, scaled by how surprising the result was. Beating a much stronger side is "
            "worth more than beating a weaker one. It gives us a principled, continuously-updated proxy for team quality that "
            "we can hold constant when asking whether travel independently matters."
        )
        st.markdown("**$R_{new} = R_{old} + K \\cdot (S - E)$** — where $K=20$, $S$ is the actual result "
                    "(1/0.5/0), and $E$ is the expected result from the rating gap.")
        st.markdown("<div class='pm-section'>Strength Leaderboard</div>", unsafe_allow_html=True)
        st.dataframe(elo_leaderboard, use_container_width=True)

        elo_long = pd.concat([
            mwe[["date", "league", "home_team", "home_elo"]].rename(columns={"home_team": "team", "home_elo": "elo"}),
            mwe[["date", "league", "away_team", "away_elo"]].rename(columns={"away_team": "team", "away_elo": "elo"}),
        ]).sort_values(["team", "date"])
        st.markdown("<div class='pm-section'>Elo Trajectories</div>", unsafe_allow_html=True)
        st.markdown("Tracing a team's Elo over time shows form arcs — rises, dynasties, and declines — that a single "
                    "season-long average would hide.")
        lg = st.selectbox("League", sorted(elo_long["league"].unique()), index=None, placeholder="Choose a league")
        if lg:
            tms = sorted(elo_long[elo_long["league"] == lg]["team"].unique())
            tm = st.selectbox("Team", tms, index=None, placeholder="Choose a team")
            if tm:
                fe = elo_long[(elo_long["league"] == lg) & (elo_long["team"] == tm)]
                fig = px.line(fe, x="date", y="elo", title=f"Elo Over Time — {tm} ({lg})",
                              labels={"elo": "Elo Rating", "date": "Date"})
                fig.update_traces(line=dict(width=1.5, color="#64ffda")); fig.update_layout(height=480, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        note(
            "<b>The key relationship:</b> if Elo tracks win percentage tightly, it confirms opponent strength is the primary "
            "determinant of results — which is exactly why it must be controlled before crediting travel with anything."
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
                "<b>Takeaway:</b> the upward trend in both panels confirms Elo captures most of what determines results. "
                "This is the confounder that the formal signal study controls for — and once it does, travel's apparent "
                "effect largely disappears."
            )
        else:
            st.info("Elo / Upsets tables unavailable.")

    with tab3:
        st.markdown(
            "An **upset** — a weaker team beating a stronger one — is where fatigue could plausibly tip the balance. "
            "Tracking upset frequency over time tests whether results are getting more or less predictable."
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
        "Comparing individual clubs makes the abstract variables concrete. Pick three teams — ideally across leagues — and "
        "see how their travel burden, rest, and away returns stack up. A team like Seattle (MLS) will show travel distances "
        "an order of magnitude larger than a London club, yet away-point returns are governed far more by quality than by "
        "those kilometres — the study's central tension, made tangible."
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
                "The raw figures for each selected club. Read across a row to compare the three teams on a single dimension; "
                "the striking gap is almost always in travel distance, while away points stay comparatively close — the "
                "visual embodiment of this study's finding that distance and results are only loosely linked."
            )
            tbl = pd.DataFrame({
                "Metric": list(metrics.keys()),
                **{nm: [round(float(r.get(c, np.nan)), 1) for c in metrics.values()] for nm, r in zip(names, rows)}
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            # --- Radar chart across normalized metrics ---
            st.markdown("<div class='pm-section'>Multi-Metric Profile</div>", unsafe_allow_html=True)
            note(
                "The radar plot normalises every metric to a 0–1 scale (each team's value relative to the maximum among the "
                "three) so that dimensions on wildly different units — a win percentage versus thousands of kilometres — can "
                "be read on one shared shape. A team that dominates a dimension reaches the outer edge; the overall silhouette "
                "gives an at-a-glance profile of each club's strengths and burdens."
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
    st.markdown("<div class='pm-section'>Findings & Formal Validation</div>", unsafe_allow_html=True)
    note(
        "<b>The headline result.</b> The descriptive views throughout this dashboard hint that travel is a weak predictor of "
        "away performance. A companion out-of-sample signal study (in the project's <code>signal_study/</code> module) tests this "
        "formally on the Premier League: using a temporal train/test split and controlling for pre-match Elo, travel distance "
        "is <b>statistically significant in-sample</b> (≈ −0.05 away-points per standard deviation of distance; p = 0.011) but "
        "adds <b>essentially zero out-of-sample predictive value</b> once opponent strength is known. In other words, the "
        "travel-fatigue effect is real but not <i>independently</i> useful for prediction — opponent quality absorbs it. "
        "Reporting that honestly, rather than overselling a weak signal, is the point of the exercise."
    )