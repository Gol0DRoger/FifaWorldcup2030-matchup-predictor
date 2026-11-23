"""Streamlit web application for the FIFA World Cup 2030 Matchup Predictor."""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="FIFA World Cup 2030 Matchup Predictor",
    page_icon="⚽",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .winner-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🏆 FIFA World Cup 2030 Matchup Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Machine learning match simulation powered by'
    " historic Elo ratings, form trajectory, and rolling performance"
    " metrics.</div>",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def fetch_api_health():
  try:
    resp = requests.get(f"{API_URL}/health", timeout=3)
    if resp.status_code == 200:
      return resp.json()
  except Exception:
    return None
  return None


@st.cache_data(ttl=300)
def fetch_teams():
  try:
    resp = requests.get(f"{API_URL}/teams", timeout=5)
    if resp.status_code == 200:
      return resp.json().get("teams", [])
  except Exception:
    return []
  return []


health_info = fetch_api_health()
teams_list = fetch_teams()

# Sidebar
with st.sidebar:
  st.header("⚙️ System Status")
  if health_info:
    st.success("● API Backend Connected")
    st.markdown(f"**Model Version:** `{health_info.get('model_version')}`")
    st.markdown(
        f"**Trained On Dataset:** `{health_info.get('trained_on_date')}`"
    )
    st.markdown(f"**Available FIFA Teams:** `{len(teams_list)}`")
  else:
    st.error("❌ Backend API Unreachable")
    st.caption(
        f"Attempted connection to `{API_URL}`. Start the FastAPI server using:"
    )
    st.code("uvicorn app.main:app --host 127.0.0.1 --port 8000")

  st.divider()
  st.markdown("### About World Cup 2030 Model")
  st.caption(
      "The FIFA World Cup 2030 marks the centenary tournament across Spain,"
      " Portugal, Morocco, Uruguay, Argentina, and Paraguay.\n\n"
      "Features are computed chronologically without future leakage from all"
      " international matches since 1872."
  )

if not teams_list:
  st.warning(
      "⚠️ Unable to load teams from the API server. Please ensure the FastAPI"
      f" backend is running on `{API_URL}`."
  )
  st.stop()

# Selection Form
col1, col2 = st.columns(2)

with col1:
  default_idx_a = (
      teams_list.index("Spain")
      if "Spain" in teams_list
      else (teams_list.index("Brazil") if "Brazil" in teams_list else 0)
  )
  team_a = st.selectbox(
      "Select Team A", options=teams_list, index=default_idx_a, key="team_a_sel"
  )

with col2:
  default_idx_b = (
      teams_list.index("France")
      if "France" in teams_list
      else (teams_list.index("Argentina") if "Argentina" in teams_list else 1)
  )
  team_b = st.selectbox(
      "Select Team B", options=teams_list, index=default_idx_b, key="team_b_sel"
  )

venue_option = st.radio(
    "Match Venue",
    options=[
        "Neutral Venue (e.g., standard World Cup match)",
        f"Team A Home ({team_a} host advantage)",
        f"Team B Home ({team_b} host advantage)",
    ],
    index=0,
    horizontal=True,
)

# Map radio selection to backend venue format
if "Neutral" in venue_option:
  venue_key = "neutral"
elif "Team A Home" in venue_option:
  venue_key = "team_a_home"
else:
  venue_key = "team_b_home"

if team_a == team_b:
  st.warning("⚠️ Please select two distinct national teams.")
  predict_disabled = True
else:
  predict_disabled = False

st.write("")
predict_button = st.button(
    "Run World Cup Simulation",
    type="primary",
    disabled=predict_disabled,
    use_container_width=True,
)

if predict_button and not predict_disabled:
  payload = {"team_a": team_a, "team_b": team_b, "venue": venue_key}

  with st.spinner(f"Simulating {team_a} vs {team_b}..."):
    try:
      response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
      if response.status_code == 200:
        res = response.json()
        p_a = res["team_a_win"]
        p_draw = res["draw"]
        p_b = res["team_b_win"]
        prediction = res["prediction"]
        confidence = res["confidence"]

        # Human-friendly winner text
        if prediction == "team_a_win":
          pred_label = f"{team_a} Victory"
          pred_detail = (
              f"The model projects {team_a} to defeat {team_b} in regular time."
          )
        elif prediction == "team_b_win":
          pred_label = f"{team_b} Victory"
          pred_detail = (
              f"The model projects {team_b} to defeat {team_a} in regular time."
          )
        else:
          pred_label = "Draw (90 Mins)"
          pred_detail = (
              "Match projected to end level in regular time (extra time /"
              " penalties needed in knockouts)."
          )

        # Highlight banner
        st.markdown(
            f"""
                <div class="winner-banner">
                    <div style="font-size: 0.9rem; letter-spacing: 0.05em; text-transform: uppercase; opacity: 0.85;">Projected Outcome</div>
                    <div style="font-size: 1.8rem; font-weight: 800; margin: 4px 0;">{pred_label}</div>
                    <div style="font-size: 1rem; opacity: 0.95;">Confidence: <strong>{confidence * 100:.1f}%</strong> | {pred_detail}</div>
                </div>
                """,
            unsafe_allow_html=True,
        )

        # Three probability cards
        c1, c2, c3 = st.columns(3)
        with c1:
          st.metric(
              label=f"🥇 {team_a} Win",
              value=f"{p_a * 100:.1f}%",
              delta="Favorite" if p_a > p_b else None,
          )
          st.progress(float(p_a))

        with c2:
          st.metric(
              label="🤝 Draw",
              value=f"{p_draw * 100:.1f}%",
          )
          st.progress(float(p_draw))

        with c3:
          st.metric(
              label=f"🥈 {team_b} Win",
              value=f"{p_b * 100:.1f}%",
              delta="Favorite" if p_b > p_a else None,
          )
          st.progress(float(p_b))

        st.caption(
            "📌 *Note: Probabilities reflect ensemble tree vote proportions"
            " from RandomForestClassifier. Venue symmetry guarantees"
            " predict(A,B) equals predict(B,A) for neutral grounds.*"
        )

      else:
        err = response.json().get("detail", "Prediction request failed.")
        st.error(f"Prediction Error ({response.status_code}): {err}")
    except Exception as exc:
      st.error(f"Failed to communicate with prediction API: {exc}")
