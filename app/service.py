"""Service layer for FIFA World Cup 2030 Matchup Prediction.

Handles model loading, feature preparation, neutral-venue symmetrical averaging,
dynamic class mapping, and inference.
"""

import os
import pickle
from typing import Any
import pandas as pd


class UnknownTeamError(Exception):
  """Raised when a requested team is not a recognized FIFA member."""

  pass


class InvalidMatchupError(Exception):
  """Raised when an invalid matchup is requested (e.g.

  same team vs itself).
  """

  pass


class PredictorService:

  def __init__(self, model_path: str | None = None):
    if model_path is None:
      base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      model_path = os.path.join(base_dir, "artifacts", "model.pkl")
    self.model_path = model_path
    self.model: Any = None
    self.classes: list[str] = []
    self.class_indices: dict[str, int] = {}
    self.latest_team_stats: dict[str, dict[str, float]] = {}
    self.feature_order: list[str] = []
    self.trained_on_date: str = "unknown"
    self.model_version: str = "unknown"
    self.valid_teams: list[str] = []
    self.valid_teams_set: set[str] = set()

  def load_model(self) -> None:
    """Loads the pickle bundle once at application startup."""
    if not os.path.exists(self.model_path):
      raise FileNotFoundError(
          f"Model bundle not found at {self.model_path}. Run train.py first."
      )

    with open(self.model_path, "rb") as f:
      bundle = pickle.load(f)

    self.model = bundle["model"]
    self.classes = list(self.model.classes_)
    # Dynamic class index mapping - never assume ordering
    self.class_indices = {c: i for i, c in enumerate(self.classes)}
    self.latest_team_stats = bundle["latest_team_stats"]
    self.feature_order = bundle["feature_order"]
    self.trained_on_date = bundle.get("trained_on_date", "unknown")
    self.model_version = bundle.get("model_version", "1.0.0")
    self.valid_teams = sorted(bundle.get("valid_teams", []))
    self.valid_teams_set = set(self.valid_teams)

  def get_health(self) -> dict[str, str]:
    return {
        "status": "healthy" if self.model is not None else "uninitialized",
        "model_version": self.model_version,
        "trained_on_date": self.trained_on_date,
    }

  def get_teams(self) -> list[str]:
    return self.valid_teams

  def _build_features_df(
      self, home_team: str, away_team: str, neutral: bool
  ) -> pd.DataFrame:
    """Builds a single-row feature DataFrame strictly matching feature_order."""
    stats_h = self.latest_team_stats.get(home_team, {})
    stats_a = self.latest_team_stats.get(away_team, {})

    r_h = stats_h.get("elo", 1500.0)
    r_a = stats_a.get("elo", 1500.0)
    bonus = 0.0 if neutral else 100.0
    eff_diff = (r_h + bonus) - r_a
    raw_diff = r_h - r_a

    h_gf = stats_h.get("rolling_gf", 1.2)
    h_ga = stats_h.get("rolling_ga", 1.2)
    a_gf = stats_a.get("rolling_gf", 1.2)
    a_ga = stats_a.get("rolling_ga", 1.2)
    gd_diff = (h_gf - h_ga) - (a_gf - a_ga)

    h_form = stats_h.get("form", 1.2)
    a_form = stats_a.get("form", 1.2)
    form_diff = h_form - a_form

    h_trend = stats_h.get("elo_trend", 0.0)
    a_trend = stats_a.get("elo_trend", 0.0)
    trend_diff = h_trend - a_trend

    # Tournament is ALWAYS set internally to FIFA World Cup (weight 60.0)
    t_weight = 60.0

    feat_dict = {
        "effective_elo_diff": eff_diff,
        "elo_diff": raw_diff,
        "elo_home": r_h,
        "elo_away": r_a,
        "neutral": 1.0 if neutral else 0.0,
        "tournament_weight": t_weight,
        "home_rolling_gf": h_gf,
        "home_rolling_ga": h_ga,
        "away_rolling_gf": a_gf,
        "away_rolling_ga": a_ga,
        "rolling_gd_diff": gd_diff,
        "home_form": h_form,
        "away_form": a_form,
        "form_diff": form_diff,
        "home_elo_trend": h_trend,
        "away_elo_trend": a_trend,
        "elo_trend_diff": trend_diff,
    }

    df = pd.DataFrame([feat_dict])[self.feature_order]
    return df

  def predict(self, team_a: str, team_b: str, venue: str = "neutral") -> dict:
    if self.model is None:
      raise RuntimeError("PredictorService model is not loaded.")

    # 1. Validation
    if team_a not in self.valid_teams_set:
      raise UnknownTeamError(
          f"Team '{team_a}' is not a recognized active FIFA national team."
      )
    if team_b not in self.valid_teams_set:
      raise UnknownTeamError(
          f"Team '{team_b}' is not a recognized active FIFA national team."
      )
    if team_a == team_b:
      raise InvalidMatchupError(
          f"Cannot predict match: team_a and team_b are the same ('{team_a}')."
      )
    if venue not in ("neutral", "team_a_home", "team_b_home"):
      raise InvalidMatchupError(
          f"Invalid venue '{venue}'. Must be 'neutral', 'team_a_home', or"
          " 'team_b_home'."
      )

    # Dynamic class index lookup
    home_idx = self.class_indices["home_win"]
    draw_idx = self.class_indices["draw"]
    away_idx = self.class_indices["away_win"]

    if venue == "neutral":
      # Symmetrical averaging across both orderings:
      # Order 1: A is designated home, B is designated away
      df1 = self._build_features_df(team_a, team_b, neutral=True)
      probs1 = self.model.predict_proba(df1)[0]

      # Order 2: B is designated home, A is designated away
      df2 = self._build_features_df(team_b, team_a, neutral=True)
      probs2 = self.model.predict_proba(df2)[0]

      # Averaging:
      # team_a_win = average of (A as home win, A as away win)
      p_a = (probs1[home_idx] + probs2[away_idx]) / 2.0
      p_draw = (probs1[draw_idx] + probs2[draw_idx]) / 2.0
      p_b = (probs1[away_idx] + probs2[home_idx]) / 2.0

    elif venue == "team_a_home":
      df = self._build_features_df(team_a, team_b, neutral=False)
      probs = self.model.predict_proba(df)[0]
      p_a = probs[home_idx]
      p_draw = probs[draw_idx]
      p_b = probs[away_idx]

    elif venue == "team_b_home":
      df = self._build_features_df(team_b, team_a, neutral=False)
      probs = self.model.predict_proba(df)[0]
      p_a = probs[away_idx]
      p_draw = probs[draw_idx]
      p_b = probs[home_idx]

    # Normalize to avoid tiny floating point drift
    total = p_a + p_draw + p_b
    p_a = round(float(p_a / total), 4)
    p_draw = round(float(p_draw / total), 4)
    p_b = round(float(p_b / total), 4)

    # Determine confidence (highest probability) and prediction label
    confidence = max(p_a, p_draw, p_b)
    if confidence == p_a:
      prediction = "team_a_win"
    elif confidence == p_b:
      prediction = "team_b_win"
    else:
      prediction = "draw"

    return {
        "team_a": team_a,
        "team_b": team_b,
        "venue": venue,
        "team_a_win": p_a,
        "draw": p_draw,
        "team_b_win": p_b,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": self.model_version,
    }
