# San Jose Sharks Performance Dashboard by Garrett Seaton Howe
# Version: 1.2
# Updated: 11/07/2025

import os
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------------------------------------------
# 🔹 App Setup
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Team Dashboard", layout="wide")

st.title("San Jose Sharks Performance Dashboard")
st.markdown("A simple, mobile-friendly dashboard built with Streamlit & Plotly.")

# --------------------------------------------------------------------------------------
# 🔹 Helper Functions
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
# 🔹 Load and Process Data
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

# Prefer position data from standard_stats if available, otherwise fall back to team_roster
if "standard_stats" in dataframes:
    primary_pos = dataframes["standard_stats"][["player", "pos"]].copy()
else:
    primary_pos = dataframes["team_roster"][["player", "pos"]].copy()

merged_df = merge_player_data(dataframes)
merged_df = merged_df.drop(columns=["pos"], errors="ignore")
merged_df = pd.merge(merged_df, primary_pos.rename(columns={"player": "Player"}), on="Player", how="left")

# Apply position name mapping globally
merged_df["pos"] = merged_df["pos"].map(position_map).fillna(merged_df["pos"])

# --------------------------------------------------------------------------------------
# 🔹 Background & Sidebar Styling
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
# 🔹 Team vs League Comparison
# --------------------------------------------------------------------------------------

if "team_stats" in dataframes and "team_analytics" in dataframes:
    team_stats_df = dataframes["team_stats"]
    team_analytics_df = dataframes["team_analytics"]

    sharks = team_stats_df.loc[team_stats_df["team"] == "San Jose Sharks"]
    league = team_stats_df.loc[team_stats_df["team"] == "League Average"]

    if sharks.empty or league.empty:
        st.warning("Team or League Average not found in team_stats.")
    else:
        team_row, league_row = sharks.iloc[0], league.iloc[0]

        # Mapping lowercase CSV keys to uppercase display names
        metric_map = {
            "W": "w",
            "L": "l",
            "OL": "ol",
            "GF/G": "gf/g",
            "GA/G": "ga/g",
            "PP%": "pp%",
            "PK%": "pk%",
            "S%": "s%",
            "SV%": "sv%",
            "PDO": "pdo"
        }

        ratio_data = []
        for display_name, csv_key in metric_map.items():
            if csv_key in team_row and csv_key in league_row:
                team_val = pd.to_numeric(team_row[csv_key], errors="coerce")
                league_val = pd.to_numeric(league_row[csv_key], errors="coerce")
                if pd.notna(team_val) and pd.notna(league_val) and league_val != 0:
                    ratio_data.append({
                        "Metric": display_name,       # show uppercase names
                        "Team Value": team_val,
                        "League Value": league_val,
                        "Ratio": team_val / league_val
                    })

        if not ratio_data:
            st.warning("No matching metrics found for comparison.")
        else:
            ratio_df = pd.DataFrame(ratio_data)
            ratio_df["Color"] = ratio_df["Ratio"].apply(lambda x: "#16a34a" if x >= 1 else "#dc2626")
            ratio_df["Label"] = ratio_df.apply(
                lambda r: f"{r['Team Value']:.2f} : {r['League Value']:.2f}", axis=1
            )

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
# 🔹 Player Roster View + Filters
# --------------------------------------------------------------------------------------

if "team_roster" not in dataframes:
    st.error("Team Roster not found in /data folder.")
    st.stop()

roster_df = normalize_columns(dataframes["team_roster"])
roster_df["player"] = roster_df["player"].astype(str).str.title()
roster_df["age"] = pd.to_numeric(roster_df["age"], errors="coerce")
roster_df["wt"] = pd.to_numeric(roster_df["wt"], errors="coerce")

# Replace abbreviated positions with full names
roster_df["pos"] = roster_df["pos"].map(position_map).fillna(roster_df["pos"])

# --- Sidebar Filters ---
st.sidebar.header("Search & Filter")

search_query = st.sidebar.text_input("Search Player", "")
st.sidebar.markdown("### Filter by Position")
positions = sorted(merged_df["pos"].dropna().unique())

selected_positions = []
for pos in positions:
    if st.sidebar.checkbox(pos, value=True):
        selected_positions.append(pos)

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

# --- Player Table (with Goals & Assists) ---
st.subheader("Roster Overview")

# Add Goals & Assists columns if present in merged_df
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
    filtered.rename(columns={
        "player": "Player",
        "age": "Age",
        "wt": "Weight",
        "no": "No."
    })[["No.", "Player", "Position", "Age", "Weight", "Goals", "Assists"]],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------------------------------------------
# 🔹 Footer
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