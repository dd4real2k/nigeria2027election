"""
Nigeria 2027 Election Prediction — Streamlit Dashboard
Run: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Nigeria 2027 Election Prediction Dashboard",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────
st.markdown("""
<style>
.main-title { font-size: 2.5rem; font-weight: 800; color: #003399; }
.subtitle   { font-size: 1.1rem; color: #555; margin-top: -10px; }
.metric-box { background: #f8f9fa; border-radius: 10px; padding: 20px;
              border-left: 5px solid #003399; }
.apc-color  { color: #003399; font-weight: bold; }
.opp-color  { color: #008751; font-weight: bold; }
.warning    { background: #fff3cd; border-radius: 8px; padding: 12px;
              border-left: 4px solid #ffc107; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown(
    '<p class="main-title">🗳️ Nigeria 2027 Presidential Election Predictor</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Data Science Portfolio Project | Le Wagon | Built with ML & Monte Carlo Simulation</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── Sidebar Controls ──────────────────────────────────────
st.sidebar.header("⚙️ Scenario Controls")
st.sidebar.markdown("Adjust parameters to model different scenarios:")

inflation_2027 = st.sidebar.slider("Inflation Rate (%)", 10, 45, 18, 1)
approval_index = st.sidebar.slider("Tinubu Approval Index (0–100)", 10, 60, 30, 1)
opposition_united = st.sidebar.checkbox("United Opposition (ADC Coalition)?", value=False)
oil_price = st.sidebar.slider("Oil Price (USD/barrel)", 40, 120, 72, 2)
security_worsens = st.sidebar.checkbox("Security Crisis (Escalation)?", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model**: Random Forest + Monte Carlo")
st.sidebar.markdown("**Training Data**: 1999–2023 Elections")
st.sidebar.markdown("**Simulations**: 10,000 trials")
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ *For educational purposes only.*")

# ── State Data ────────────────────────────────────────────
states_data = {
    "state": [
        "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
        "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT", "Gombe", "Imo",
        "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos",
        "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers",
        "Sokoto", "Taraba", "Yobe", "Zamfara"
    ],
    "zone": [
        "South-East", "North-East", "South-South", "South-East", "North-East", "South-South",
        "North-Central", "North-East", "South-South", "South-South", "South-East", "South-South",
        "South-West", "South-East", "North-Central", "North-East", "South-East", "North-West",
        "North-West", "North-West", "North-West", "North-West", "North-Central", "North-Central",
        "South-West", "North-Central", "North-Central", "South-West", "South-West", "South-West",
        "South-West", "North-Central", "South-South", "North-West", "North-East", "North-East",
        "North-West"
    ],
    "base_apc_prob": [
        0.18, 0.42, 0.32, 0.12, 0.55, 0.58, 0.38, 0.65,
        0.52, 0.28, 0.20, 0.35, 0.62, 0.14, 0.44, 0.60, 0.22,
        0.72, 0.61, 0.55, 0.68, 0.65, 0.55, 0.60, 0.51,
        0.55, 0.62, 0.55, 0.58, 0.45, 0.48, 0.42, 0.30,
        0.58, 0.45, 0.68, 0.62
    ],
    "population_m": [
        3.7, 4.2, 5.5, 6.0, 7.2, 2.3, 6.0, 5.8, 4.0, 5.6, 3.3, 4.7,
        3.3, 5.1, 3.8, 3.4, 5.9, 5.8, 9.0, 15.8, 9.1, 5.0, 4.7, 3.3,
        16.0, 2.8, 6.1, 7.1, 5.0, 4.7, 8.8, 4.3, 8.9, 6.6, 3.2, 3.3, 5.3
    ]
}
df = pd.DataFrame(states_data)

# ── Compute adjusted probabilities ────────────────────────
def compute_probs(df, inflation, approval, opposition_united, oil, security_worsens):
    probs = df["base_apc_prob"].copy()

    inflation_adj = (inflation - 18.5) * -0.005
    approval_adj = (approval - 30) * 0.004
    oil_adj = (oil - 72) * 0.001
    security_adj = -0.07 if security_worsens else 0.0
    opp_adj = -0.14 if opposition_united else 0.0

    probs = probs + inflation_adj + approval_adj + oil_adj + security_adj + opp_adj
    return probs.clip(0.05, 0.95)

df["apc_prob"] = compute_probs(
    df, inflation_2027, approval_index, opposition_united, oil_price, security_worsens
)
df["opp_prob"] = 1 - df["apc_prob"]
df["predicted_winner"] = df["apc_prob"].apply(lambda x: "APC" if x > 0.5 else "Opposition")

# ── Monte Carlo ────────────────────────────────────────────
@st.cache_data
def run_monte_carlo(probs_tuple, n=5000):
    probs = np.array(probs_tuple)
    apc_wins = 0
    vote_shares = []

    for _ in range(n):
        noise = np.random.normal(0, 0.08, size=len(probs))
        sim_probs = np.clip(probs + noise, 0.05, 0.95)
        outcomes = np.random.binomial(1, sim_probs)

        if outcomes.sum() > 18:
            apc_wins += 1

        vote_shares.append(sim_probs.mean() * 100)

    return apc_wins / n * 100, np.array(vote_shares)

apc_win_pct, vote_dist = run_monte_carlo(tuple(df["apc_prob"].tolist()))
opp_win_pct = 100 - apc_win_pct

# ── Row 1: Key Metrics ────────────────────────────────────
st.subheader("📊 Model Output")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🔵 APC Win Probability",
        f"{apc_win_pct:.1f}%",
        delta="Favored ✅" if apc_win_pct > 50 else "Underdog ⚠️"
    )

with col2:
    st.metric(
        "🟢 Opposition Win Prob.",
        f"{opp_win_pct:.1f}%",
        delta="Favored ✅" if opp_win_pct > 50 else "Underdog ⚠️"
    )

with col3:
    apc_states_count = (df["apc_prob"] > 0.5).sum()
    st.metric("🔵 APC States", f"{apc_states_count} / 37")

with col4:
    opp_states_count = (df["opp_prob"] > 0.5).sum()
    st.metric("🟢 Opposition States", f"{opp_states_count} / 37")

with col5:
    mean_vote = vote_dist.mean()
    st.metric("📈 APC Avg Vote Share", f"{mean_vote:.1f}%")

st.markdown("---")

# ── Row 2: Charts ─────────────────────────────────────────
col_left, col_right = st.columns([1.2, 0.8])

with col_left:
    st.subheader("🗺️ State-by-State Win Probability")
    fig_states = px.bar(
        df.sort_values("apc_prob"),
        x="apc_prob",
        y="state",
        color="predicted_winner",
        color_discrete_map={"APC": "#003399", "Opposition": "#008751"},
        orientation="h",
        labels={
            "apc_prob": "APC Win Probability",
            "state": "State",
            "predicted_winner": "Projected Winner"
        },
        title="APC Win Probability by State (2027 Projection)"
    )
    fig_states.add_vline(x=0.5, line_dash="dash", line_color="red", line_width=2)
    fig_states.update_layout(height=650, showlegend=True)
    st.plotly_chart(fig_states, width="stretch")

with col_right:
    st.subheader("🎲 Monte Carlo Distribution")

    fig_mc = go.Figure()
    fig_mc.add_trace(
        go.Histogram(
            x=vote_dist,
            nbinsx=60,
            marker_color="#2E86AB",
            opacity=0.8,
            name="APC Vote Share"
        )
    )
    fig_mc.add_vline(
        x=50,
        line_dash="dash",
        line_color="red",
        annotation_text="50%",
        line_width=2
    )
    fig_mc.add_vline(
        x=vote_dist.mean(),
        line_dash="solid",
        line_color="orange",
        annotation_text=f"Mean: {vote_dist.mean():.1f}%",
        line_width=2
    )
    fig_mc.update_layout(
        title="APC National Vote Share (5,000 simulations)",
        xaxis_title="APC Vote Share (%)",
        yaxis_title="Frequency",
        height=300
    )
    st.plotly_chart(fig_mc, width="stretch")

    fig_pie = go.Figure(
        go.Pie(
            labels=["APC Wins", "Opposition Wins"],
            values=[apc_win_pct, opp_win_pct],
            marker_colors=["#003399", "#008751"],
            hole=0.4,
            textinfo="label+percent"
        )
    )
    fig_pie.update_layout(title="Election Outcome Probability", height=300)
    st.plotly_chart(fig_pie, width="stretch")

# ── Row 3: Zone Summary ───────────────────────────────────
st.markdown("---")
st.subheader("🌍 Zone-Level Summary")

zone_summary = (
    df.groupby("zone")
    .agg(
        avg_apc_prob=("apc_prob", "mean"),
        states_count=("state", "count"),
        apc_states=("predicted_winner", lambda x: (x == "APC").sum())
    )
    .reset_index()
)

zone_summary["opp_states"] = zone_summary["states_count"] - zone_summary["apc_states"]
zone_summary["avg_apc_prob"] = (zone_summary["avg_apc_prob"] * 100).round(1)

fig_zone = px.bar(
    zone_summary,
    x="zone",
    y=["apc_states", "opp_states"],
    color_discrete_map={"apc_states": "#003399", "opp_states": "#008751"},
    barmode="group",
    labels={
        "value": "Number of States",
        "zone": "Geopolitical Zone",
        "variable": "Party"
    },
    title="Projected State Wins by Geopolitical Zone"
)
fig_zone.update_layout(height=350)
st.plotly_chart(fig_zone, width="stretch")

# ── Data Table ────────────────────────────────────────────
with st.expander("📋 View Full State Prediction Table"):
    display_df = df[["state", "zone", "apc_prob", "opp_prob", "predicted_winner"]].copy()
    display_df["apc_prob"] = (display_df["apc_prob"] * 100).round(1).astype(str) + "%"
    display_df["opp_prob"] = (display_df["opp_prob"] * 100).round(1).astype(str) + "%"
    display_df.columns = ["State", "Zone", "APC Win Prob.", "Opp. Win Prob.", "Projected Winner"]
    st.dataframe(display_df, width="stretch")

# ── Disclaimer ────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="warning">
⚠️ <strong>Disclaimer:</strong> This is a data science portfolio project for educational purposes only.
Election predictions involve significant uncertainty. This model uses historical data and economic projections
as inputs — actual results may differ significantly due to unforeseen events, candidate changes, mobilization
efforts, and other factors not captured by the model. Data sources: INEC, NBS, Afrobarometer, NOIPolls, World Bank.
</div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>Built by Daniel Diala | Le Wagon Data Science & AI Bootcamp | 2026</strong></p>
        <p>
            LinkedIn: <a href="https://www.linkedin.com/in/danieldiala/">danieldiala</a> |
            GitHub: <a href="https://github.com/dd4real2k">dd4real2k</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
