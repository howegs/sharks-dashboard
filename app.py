# San Jose Sharks Performance Dashboard by Garrett Seaton Howe
# Version: 1.3
# Updated: 11/07/2025

import os
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------------------------------------------
#  Header
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Team Dashboard", layout="wide")
st.title("San Jose Sharks Performance Dashboard")
st.markdown("A simple, mobile-friendly dashboard built with Streamlit & Plotly.")
st.markdown("Data Updated: 11/05/2025")

# --------------------------------------------------------------------------------------
#  Background & Styling
# --------------------------------------------------------------------------------------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://news.sportslogos.net/wp-content/uploads/2016/08/Sharks-New-Logo.png");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(15, 32, 45, 0.7);
    z-index: 0;
}
[data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }
[data-testid="stHeader"] h1 { color: #FFF; text-shadow: 1px 1px 3px #000; }
[data-testid="stSidebar"] {
    background-color: #006D75 !important;
    color: #C4CED4 !important;
}
[data-testid="stSidebar"] input[type="text"] {
    background-color: #003B41 !important;
    color: #FFF !important;
    border-radius: 8px !important;
    border: 1px solid #E57200 !important;
    padding: 6px 8px !important;
}
[data-baseweb="slider"] .st-bd { background-color: #E57200 !important; }
[data-baseweb="slider"] [role="slider"] {
    background-color: #E57200 !important;
    border: 2px solid #FFF !important;
    border-radius: 50% !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
#  Helper Functions to pull in CSVs and merge them into a single DataFrame
# --------------------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for consistency."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )
    return df

@st.cache_data
def load_all_csvs(folder_path="data"):
    """Loads all CSV files in a folder into a dictionary of DataFrames."""
    dataframes = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            key = os.path.splitext(filename)[0]
            try:
                if filename == "game_stats.csv":
                    df = pd.read_csv(os.path.join(folder_path, filename), dtype=str)
                else:
                    df = pd.read_csv(os.path.join(folder_path, filename))
                    df = normalize_columns(df)
                    if "pos" in df.columns:
                        df["pos"] = df["pos"].astype(str).str.strip().str.lower()
                dataframes[key] = df
            except Exception as e:
                st.warning(f"⚠️ Could not load {filename}: {e}")
    if not dataframes:
        st.error("❌ No data found in folder.")
    else:
        st.success(f"✅ Loaded {len(dataframes)} CSV files successfully.")
    return dataframes

@st.cache_data
def merge_player_data(dataframes):
    """Merge all CSVs containing a 'Player' column."""
    dfs_with_player = []
    for name, df in dataframes.items():
        df = df.copy()
        if "player" in df.columns:
            df.rename(columns={"player": "Player"}, inplace=True)
            dfs_with_player.append(df)
    if not dfs_with_player:
        return next(iter(dataframes.values()))

    merged_df = dfs_with_player[0].copy()
    for df in dfs_with_player[1:]:
        merged_df = pd.merge(
            merged_df, df, on="Player", how="outer", suffixes=("", "_dup"), validate="1:1"
        )
        dup_cols = [c for c in merged_df.columns if c.endswith("_dup")]
        for dc in dup_cols:
            orig = dc[:-4]
            merged_df[orig] = merged_df[orig].combine_first(merged_df[dc])
            merged_df.drop(columns=[dc], inplace=True)

    return merged_df.drop_duplicates(subset=["Player"], keep="first")

# --------------------------------------------------------------------------------------
#  Load Data
# --------------------------------------------------------------------------------------
dataframes = load_all_csvs("data")
merged_df = merge_player_data(dataframes)

# Map shorthand positions to full names
position_map = {
    "d": "Defense",
    "f": "Forward",
    "lw": "Left Wing",
    "rw": "Right Wing",
    "c": "Captain",
    "g": "Goalie"
}

# Prefer position data from standard_stats if available
if "standard_stats" in dataframes:
    primary_pos = dataframes["standard_stats"][["player", "pos"]].copy()
else:
    primary_pos = dataframes["team_roster"][["player", "pos"]].copy()

merged_df = merged_df.drop(columns=["pos"], errors="ignore")
merged_df = pd.merge(merged_df, primary_pos.rename(columns={"player": "Player"}), on="Player", how="left")
merged_df["pos"] = merged_df["pos"].map(position_map).fillna(merged_df["pos"])


# --------------------------------------------------------------------------------------
#  Game Stats Display Function
# --------------------------------------------------------------------------------------
def display_game_stats(df: pd.DataFrame):
    """Display recent and full season games preserving original CSV headers."""
    if df.empty:
        st.warning("Game stats not available yet (no completed games found).")
        return

    raw_games = df.copy()
    working_date = pd.to_datetime(raw_games.get("Date", pd.Series([pd.NaT]*len(raw_games))), errors="coerce")

    def find_column(candidates):
        for c in candidates:
            if c in raw_games.columns:
                return c
        return None

    gp_col       = find_column(["GP", "Gp", "gp"])
    date_col     = find_column(["Date", "date"])
    opponent_col = find_column(["Opponent", "opponent", "Opp"])
    gf_col       = find_column(["GF", "Gf", "gf", "GF/G"])
    ga_col       = find_column(["GA", "Ga", "ga"])
    w_col        = find_column(["W", "w"])
    l_col        = find_column(["L", "l"])
    streak_col   = find_column(["Streak", "streak"])

    gf_series = pd.to_numeric(raw_games[gf_col], errors="coerce") if gf_col else pd.Series([pd.NA]*len(raw_games))
    ga_series = pd.to_numeric(raw_games[ga_col], errors="coerce") if ga_col else pd.Series([pd.NA]*len(raw_games))

    helper = pd.DataFrame({
        "_idx": raw_games.index,
        "_date": working_date,
        "_gf": gf_series,
        "_ga": ga_series,
        "_gp": raw_games[gp_col] if gp_col else pd.NA,
        "_opp": raw_games[opponent_col] if opponent_col else pd.NA,
        "_w_flag": raw_games[w_col] if w_col else pd.NA,
        "_l_flag": raw_games[l_col] if l_col else pd.NA,
        "_streak": raw_games[streak_col] if streak_col else pd.NA,
    })

    played_mask = helper["_gf"].notna() & helper["_ga"].notna()
    played = helper[played_mask].sort_values("_date", ascending=False)

    def deduce_result(row):
        if pd.notna(row["_w_flag"]) and str(row["_w_flag"]).strip(): return str(row["_w_flag"]).strip()
        if pd.notna(row["_l_flag"]) and str(row["_l_flag"]).strip(): return str(row["_l_flag"]).strip()
        try:
            if row["_gf"] > row["_ga"]: return "W"
            elif row["_gf"] < row["_ga"]: return "L"
            else: return "T"
        except: return ""
    played["_result"] = played.apply(deduce_result, axis=1)

    # --- Recent Games ---
    recent = played.head(3)
    recent_display = pd.DataFrame()
    if date_col:     recent_display[date_col] = raw_games.loc[recent["_idx"], date_col].values
    if opponent_col: recent_display[opponent_col] = raw_games.loc[recent["_idx"], opponent_col].values
    if gf_col:       recent_display[gf_col] = raw_games.loc[recent["_idx"], gf_col].values
    if ga_col:       recent_display[ga_col] = raw_games.loc[recent["_idx"], ga_col].values
    result_label = w_col if w_col else "Result"
    recent_display[result_label] = played.loc[recent.index, "_result"].values
    recent_display.columns = [c.capitalize() for c in recent_display.columns]

    # --- Stylized HTML table for Recent Games ---
    def style_recent_games(df):
        html = '<table style="width:100%; border-collapse: collapse; text-align:center;">'
        html += '<tr style="background-color:#006D75; color:#FFF; font-weight:bold;">'
        for col in df.columns:
            html += f'<th style="padding:6px 10px; border: 1px solid #E57200;">{col}</th>'
        html += '</tr>'
        for i, row in df.iterrows():
            bg = "#003B41" if i % 2 == 0 else "#004C55"
            html += f'<tr style="background-color:{bg}; color:#FFF;">'
            for col in df.columns:
                val = row[col]
                if col in [gf_col.capitalize(), ga_col.capitalize()]:
                    val = f'<b style="color:#E57200;">{val}</b>'
                html += f'<td style="padding:6px 10px; border: 1px solid #E57200;">{val}</td>'
            html += '</tr>'
        html += '</table>'
        return html

    st.markdown("## 🏒 Recent Games", unsafe_allow_html=True)
    st.markdown(style_recent_games(recent_display), unsafe_allow_html=True)

    # --- Full Season ---
    full_display = pd.DataFrame()
    if gp_col:       full_display[gp_col] = raw_games.loc[played["_idx"], gp_col].values
    if date_col:     full_display[date_col] = raw_games.loc[played["_idx"], date_col].values
    if opponent_col: full_display[opponent_col] = raw_games.loc[played["_idx"], opponent_col].values
    if gf_col:       full_display[gf_col] = raw_games.loc[played["_idx"], gf_col].values
    if ga_col:       full_display[ga_col] = raw_games.loc[played["_idx"], ga_col].values
    full_display[result_label] = played["_result"].values
    if streak_col:   full_display[streak_col] = raw_games.loc[played["_idx"], streak_col].values
    full_display.columns = [c.capitalize() for c in full_display.columns]

    st.markdown("### 📅 Full Season Game Results")
    st.dataframe(full_display, hide_index=True, use_container_width=True)
    


# --------------------------------------------------------------------------------------
#  Display Recent Games at Top
# --------------------------------------------------------------------------------------
if "game_stats" in dataframes:
    display_game_stats(dataframes["game_stats"])
else:
    st.warning("Game stats CSV not found in /data folder.")

# --------------------------------------------------------------------------------------
#  Team vs League Comparison
# --------------------------------------------------------------------------------------
if "team_stats" in dataframes and "team_analytics" in dataframes:
    team_stats_df = dataframes["team_stats"]
    sharks = team_stats_df.loc[team_stats_df["team"] == "San Jose Sharks"]
    league = team_stats_df.loc[team_stats_df["team"] == "League Average"]

    if sharks.empty or league.empty:
        st.warning("Team or League Average not found in team_stats.")
    else:
        team_row, league_row = sharks.iloc[0], league.iloc[0]
        metric_map = {
            "W": "w", "L": "l", "OL": "ol", "GF/G": "gf/g", "GA/G": "ga/g",
            "PP%": "pp%", "PK%": "pk%", "S%": "s%", "SV%": "sv%", "PDO": "pdo"
        }

        ratio_data = []
        for display_name, csv_key in metric_map.items():
            if csv_key in team_row and csv_key in league_row:
                team_val = pd.to_numeric(team_row[csv_key], errors="coerce")
                league_val = pd.to_numeric(league_row[csv_key], errors="coerce")
                if pd.notna(team_val) and pd.notna(league_val) and league_val != 0:
                    ratio_data.append({
                        "Metric": display_name,
                        "Team Value": team_val,
                        "League Value": league_val,
                        "Ratio": team_val / league_val
                    })

        if ratio_data:
            ratio_df = pd.DataFrame(ratio_data)
            ratio_df["Color"] = ratio_df["Ratio"].apply(lambda x: "#16a34a" if x >= 1 else "#dc2626")
            ratio_df["Label"] = ratio_df.apply(lambda r: f"{r['Team Value']:.2f} : {r['League Value']:.2f}", axis=1)
            fig_ratio = px.bar(
                ratio_df, x="Ratio", y="Metric", orientation="h",
                text="Label", color="Color", color_discrete_map="identity"
            )
            fig_ratio.update_traces(textposition="outside")
            fig_ratio.update_layout(
                title="🏒 Team Record vs League Average (Sharks : League)",
                xaxis_title="Ratio to League Average",
                yaxis_title="Metric",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=60, r=40, t=70, b=50)
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
else:
    st.warning("Missing required files: team_stats or team_analytics")

# --------------------------------------------------------------------------------------
#  Player Roster View + Filters
# --------------------------------------------------------------------------------------
if "team_roster" not in dataframes:
    st.error("Team Roster not found in /data folder.")
    st.stop()

roster_df = normalize_columns(dataframes["team_roster"])
roster_df["player"] = roster_df["player"].astype(str).str.title()
roster_df["age"] = pd.to_numeric(roster_df["age"], errors="coerce")
roster_df["wt"] = pd.to_numeric(roster_df["wt"], errors="coerce")
roster_df["pos"] = roster_df["pos"].map(position_map).fillna(roster_df["pos"])

# --- Sidebar Filters ---
st.sidebar.header("Search & Filter")
search_query = st.sidebar.text_input("Search Player", "")
st.sidebar.markdown("### Filter by Position")
positions = sorted(merged_df["pos"].dropna().unique())
selected_positions = [pos for pos in positions if st.sidebar.checkbox(pos, value=True)]

age_range = st.sidebar.slider(
    "Filter by Age",
    min_value=int(roster_df["age"].min()),
    max_value=int(roster_df["age"].max()),
    value=(int(roster_df["age"].min()), int(roster_df["age"].max()))
)

# Apply filters
filtered = roster_df.copy()
if selected_positions:
    filtered = filtered[filtered["pos"].isin(selected_positions)]
filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]
if search_query:
    filtered = filtered[filtered["player"].str.contains(search_query, case=False, na=False)]

# --- KPI Summary ---
col1, col2, col3 = st.columns(3)
col3.metric("Average Age", f"{filtered['age'].mean():.1f}")
col2.metric("Average Weight", f"{filtered['wt'].mean():.1f} lbs")
col1.metric("Players Displayed", len(filtered))

# --- Player Table ---
st.subheader("Roster Overview")
if "g" in merged_df.columns and "a" in merged_df.columns:
    stats_df = merged_df[["Player", "g", "a", "pos"]].rename(
        columns={"g": "Goals", "a": "Assists", "pos": "Position"}
    )
    filtered = filtered.merge(stats_df, left_on="player", right_on="Player", how="left")
    filtered.drop(columns=["Player"], inplace=True)
else:
    filtered["Goals"] = None
    filtered["Assists"] = None
    filtered["Position"] = filtered["pos"].map(position_map).fillna(filtered["pos"])

st.dataframe(
    filtered.rename(columns={"player": "Player", "age": "Age", "wt": "Weight", "no": "No."})[
        ["No.", "Player", "Position", "Age", "Weight", "Goals", "Assists"]
    ],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------------------------------------------
#  Footer
# --------------------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    """
    <div style="
        text-align: center;
        color: #EEE;
        font-size: 0.9em;
        background: rgba(0, 109, 117, 0.6);
        padding: 8px;
        border-radius: 10px;
        width: 90%;
        margin: auto;
        box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
    ">
    Data Source: Hockey-Reference.com — Background Image: SportsLogos.net<br>
    Dashboard built with <a href="https://streamlit.io" target="_blank" style="color:#E57200;">Streamlit</a> & 
    <a href="https://plotly.com/python/" target="_blank" style="color:#E57200;">Plotly</a>.
    </div>
    """,
    unsafe_allow_html=True
)