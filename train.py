"""
FIFA World Cup 2030 Matchup Predictor - Training & Evaluation Pipeline
Computes Elo ratings and historical rolling features strictly chronologically without data leakage.
Trains a RandomForestClassifier alongside two baselines and exports artifacts/model.pkl.
"""

from collections import defaultdict
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "results.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PKL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")


def get_tournament_weight(tournament: str) -> float:
  """Assign tournament importance weights following World Football Elo conventions:

  - FIFA World Cup finals: 60
  - Continental finals: 50
  - World Cup & Continental Qualifiers: 40
  - Other official tournaments (Nations League, etc.): 30
  - Friendlies & minor tournaments: 20
  """
  tl = tournament.lower()
  if "fifa world cup" in tl and "qualification" not in tl:
    return 60.0
  elif any(
      c in tl
      for c in [
          "uefa euro",
          "copa américa",
          "copa america",
          "african cup of nations",
          "afc asian cup",
          "gold cup",
          "confederations cup",
      ]
  ) and not any(q in tl for q in ["qualification", "preliminary"]):
    return 50.0
  elif any(
      q in tl for q in ["qualification", "qualifying", "preliminary", "play-off"]
  ):
    return 40.0
  elif any(
      n in tl
      for n in [
          "nations league",
          "cecafa",
          "cosafa",
          "gulf cup",
          "aff",
          "caribbean cup",
          "arab cup",
      ]
  ):
    return 30.0
  else:
    return 20.0


def extract_features_and_elo(df: pd.DataFrame):
  """Iterates through matches chronologically from 1872 to warm up Elo ratings.

  Computes pre-match features BEFORE each match (strictly no leakage).
  Updates team ratings and histories AFTER computing features.
  """
  df = df.copy()
  df["date"] = pd.to_datetime(df["date"])
  df = df.sort_values("date", kind="stable").reset_index(drop=True)

  elos = defaultdict(lambda: 1500.0)
  elo_history = defaultdict(list)
  gf_history = defaultdict(list)
  ga_history = defaultdict(list)
  pts_history = defaultdict(list)

  features = []
  targets = []
  meta = []

  print(f"Extracting features across {len(df)} international matches...")

  for idx, row in df.iterrows():
    h = row["home_team"]
    a = row["away_team"]
    hs = int(row["home_score"])
    as_ = int(row["away_score"])
    neutral = bool(row["neutral"])
    date = row["date"]
    tournament = row["tournament"]

    # 1. Team states BEFORE this match
    r_h = elos[h]
    r_a = elos[a]
    bonus = 0.0 if neutral else 100.0
    eff_diff = (r_h + bonus) - r_a
    raw_diff = r_h - r_a

    # Rolling goals (last 10 matches)
    h_gf = float(np.mean(gf_history[h][-10:])) if gf_history[h] else 1.2
    h_ga = float(np.mean(ga_history[h][-10:])) if ga_history[h] else 1.2
    a_gf = float(np.mean(gf_history[a][-10:])) if gf_history[a] else 1.2
    a_ga = float(np.mean(ga_history[a][-10:])) if ga_history[a] else 1.2
    gd_diff = (h_gf - h_ga) - (a_gf - a_ga)

    # Recent form: points per match over last 5 matches (W=3, D=1, L=0)
    h_form = float(np.mean(pts_history[h][-5:])) if pts_history[h] else 1.2
    a_form = float(np.mean(pts_history[a][-5:])) if pts_history[a] else 1.2
    form_diff = h_form - a_form

    # Elo trend: change over last 5 matches
    h_trend = (
        float(r_h - elo_history[h][-5])
        if len(elo_history[h]) >= 5
        else float(r_h - elo_history[h][0] if elo_history[h] else 0.0)
    )
    a_trend = (
        float(r_a - elo_history[a][-5])
        if len(elo_history[a]) >= 5
        else float(r_a - elo_history[a][0] if elo_history[a] else 0.0)
    )
    trend_diff = h_trend - a_trend

    t_weight = get_tournament_weight(tournament)

    feat = {
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

    target = "home_win" if hs > as_ else ("draw" if hs == as_ else "away_win")
    features.append(feat)
    targets.append(target)
    meta.append({
        "date": date,
        "tournament": tournament,
        "home_team": h,
        "away_team": a,
    })

    # 2. Update state AFTER computing pre-match features
    # Expected outcome
    e_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + bonus)) / 400.0))
    e_a = 1.0 - e_h

    # Actual outcome
    s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
    s_a = 1.0 - s_h

    # Goal difference multiplier
    diff_g = abs(hs - as_)
    if diff_g <= 1:
      g = 1.0
    elif diff_g == 2:
      g = 1.5
    else:
      g = (11.0 + diff_g) / 8.0

    k = t_weight
    delta_h = k * g * (s_h - e_h)
    delta_a = k * g * (s_a - e_a)

    # Histories
    elo_history[h].append(r_h)
    elo_history[a].append(r_a)
    gf_history[h].append(hs)
    ga_history[h].append(as_)
    gf_history[a].append(as_)
    ga_history[a].append(hs)

    h_pts = 3 if hs > as_ else (1 if hs == as_ else 0)
    a_pts = 3 if as_ > hs else (1 if hs == as_ else 0)
    pts_history[h].append(h_pts)
    pts_history[a].append(a_pts)

    # New ratings
    elos[h] = r_h + delta_h
    elos[a] = r_a + delta_a

  X_df = pd.DataFrame(features)
  y_s = pd.Series(targets)
  meta_df = pd.DataFrame(meta)

  # Compute latest per-team stats as of the end of the dataset
  latest_team_stats = {}
  all_teams = set(df["home_team"]).union(set(df["away_team"]))
  for team in all_teams:
    cur_elo = float(elos[team])
    last_gf = (
        float(np.mean(gf_history[team][-10:]))
        if gf_history[team]
        else 1.2
    )
    last_ga = (
        float(np.mean(ga_history[team][-10:]))
        if ga_history[team]
        else 1.2
    )
    last_form = (
        float(np.mean(pts_history[team][-5:]))
        if pts_history[team]
        else 1.2
    )
    last_trend = (
        float(cur_elo - elo_history[team][-5])
        if len(elo_history[team]) >= 5
        else float(cur_elo - elo_history[team][0] if elo_history[team] else 0.0)
    )

    latest_team_stats[team] = {
        "elo": cur_elo,
        "rolling_gf": last_gf,
        "rolling_ga": last_ga,
        "form": last_form,
        "elo_trend": last_trend,
    }

  return X_df, y_s, meta_df, latest_team_stats


def get_valid_fifa_teams(df: pd.DataFrame) -> list[str]:
  """Filter to real FIFA national teams:

  - Must appear in World Cup or World Cup qualification matches.
  - Must have played at least one international match since 2022-01-01.
  This filters out defunct nations, regional teams, and non-FIFA sides.
  """
  wc_tournaments = ["FIFA World Cup", "FIFA World Cup qualification"]
  wc_teams = set(
      df[df["tournament"].isin(wc_tournaments)]["home_team"]
  ).union(set(df[df["tournament"].isin(wc_tournaments)]["away_team"]))

  recent_teams = set(df[df["date"] >= "2022-01-01"]["home_team"]).union(
      set(df[df["date"] >= "2022-01-01"]["away_team"])
  )

  valid = sorted(list(wc_teams.intersection(recent_teams)))
  return valid


def train_and_evaluate():
  print("Loading results.csv...")
  df = pd.read_csv(DATA_PATH)
  valid_teams = get_valid_fifa_teams(df)
  print(f"Identified {len(valid_teams)} valid modern FIFA national teams.")

  X_df, y_s, meta_df, latest_team_stats = extract_features_and_elo(df)

  feature_order = list(X_df.columns)
  print(f"Features ({len(feature_order)}):", feature_order)

  # Time-based splits
  # Training on ALL matches from 1990 up to end of 2021
  train_mask = (meta_df["date"] >= "1990-01-01") & (
      meta_df["date"] < "2022-01-01"
  )
  test_mask = meta_df["date"] >= "2022-01-01"
  wc2022_mask = (meta_df["tournament"] == "FIFA World Cup") & (
      meta_df["date"].dt.year == 2022
  )
  wc2026_mask = (meta_df["tournament"] == "FIFA World Cup") & (
      meta_df["date"].dt.year == 2026
  )

  X_train, y_train = X_df[train_mask], y_s[train_mask]
  X_test, y_test = X_df[test_mask], y_s[test_mask]

  classes = sorted(list(y_train.unique()))  # ['away_win', 'draw', 'home_win']
  print(f"Classes: {classes}")
  print(f"Train matches (1990-2021): {len(X_train)}")
  print(f"Test matches (2022-2026): {len(X_test)}")
  print(f"2022 World Cup holdout: {wc2022_mask.sum()} matches")
  print(f"2026 World Cup holdout: {wc2026_mask.sum()} matches")

  # 1. Baseline 1: Always predict home win
  # Probability baseline: class distribution in training data
  priors = y_train.value_counts(normalize=True)[classes].values

  # 2. Baseline 2: Elo-only Logistic Regression
  lr = LogisticRegression(random_state=42, max_iter=1000)
  lr.fit(X_train[["effective_elo_diff"]], y_train)

  # 3. Main Model: RandomForestClassifier
  rf = RandomForestClassifier(
      n_estimators=250,
      max_depth=10,
      min_samples_leaf=25,
      random_state=42,
      n_jobs=-1,
  )
  rf.fit(X_train, y_train)

  # Evaluation
  eval_splits = [
      ("Test Set (2022-2026)", test_mask),
      ("Holdout: 2022 World Cup", wc2022_mask),
      ("Holdout: 2026 World Cup", wc2026_mask),
  ]

  metrics_report = {}

  print("\n" + "=" * 70)
  print("MODEL EVALUATION & HONEST BASELINE COMPARISON")
  print("=" * 70)

  for split_name, mask in eval_splits:
    X_sub = X_df[mask]
    y_sub = y_s[mask]
    n_samples = len(y_sub)

    # Baseline 1: Always Home Win
    b1_preds = ["home_win"] * n_samples
    b1_probs = np.tile(priors, (n_samples, 1))
    b1_acc = float(accuracy_score(y_sub, b1_preds))
    b1_ll = float(log_loss(y_sub, b1_probs, labels=classes))

    # Baseline 2: Elo LogReg
    b2_preds = lr.predict(X_sub[["effective_elo_diff"]])
    b2_probs = lr.predict_proba(X_sub[["effective_elo_diff"]])
    b2_acc = float(accuracy_score(y_sub, b2_preds))
    b2_ll = float(log_loss(y_sub, b2_probs, labels=classes))

    # Random Forest
    rf_preds = rf.predict(X_sub)
    rf_probs = rf.predict_proba(X_sub)
    rf_acc = float(accuracy_score(y_sub, rf_preds))
    rf_ll = float(log_loss(y_sub, rf_probs, labels=classes))

    metrics_report[split_name] = {
        "n_samples": n_samples,
        "baseline_always_home": {"accuracy": b1_acc, "log_loss": b1_ll},
        "baseline_elo_logreg": {"accuracy": b2_acc, "log_loss": b2_ll},
        "random_forest": {"accuracy": rf_acc, "log_loss": rf_ll},
    }

    print(f"\n--- {split_name} (N = {n_samples}) ---")
    print(
        f"{'Model':<28} | {'Accuracy':<10} | {'Log Loss':<10} | vs Home Baseline"
    )
    print("-" * 65)
    print(
        f"{'1. Always Home Win':<28} | {b1_acc * 100:>8.2f}% | {b1_ll:>10.4f} |"
        " (reference)"
    )
    print(
        f"{'2. Elo Logistic Regression':<28} | {b2_acc * 100:>8.2f}% |"
        f" {b2_ll:>10.4f} | +{(b2_acc - b1_acc) * 100:.2f}%"
    )
    print(
        f"{'3. Random Forest (Multi-feat)':<28} | {rf_acc * 100:>8.2f}% |"
        f" {rf_ll:>10.4f} | +{(rf_acc - b1_acc) * 100:.2f}%"
    )

  # Permutation Importance on Test Set
  print("\nComputing Permutation Feature Importance on Test Set...")
  perm = permutation_importance(
      rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
  )

  perm_data = []
  for f_name, mean, std in zip(
      feature_order, perm.importances_mean, perm.importances_std
  ):
    perm_data.append({
        "feature": f_name,
        "importance_mean": float(mean),
        "importance_std": float(std),
    })

  perm_data.sort(key=lambda x: x["importance_mean"], reverse=True)

  print("\nPermutation Feature Importance (mean drop in accuracy):")
  print(f"{'Feature':<25} | {'Importance':<12} | {'Std Dev':<10}")
  print("-" * 52)
  for item in perm_data:
    print(
        f"{item['feature']:<25} | {item['importance_mean']:>11.5f}  |"
        f" {item['importance_std']:>9.5f}"
    )

  # Refit the same RandomForest on ALL matches from 1990-01-01 through latest date
  full_train_mask = meta_df["date"] >= "1990-01-01"
  X_full, y_full = X_df[full_train_mask], y_s[full_train_mask]
  print(
      f"\nRefitting production RandomForest on ALL matches from 1990 to latest ({len(X_full)} matches)..."
  )

  rf_final = RandomForestClassifier(
      n_estimators=250,
      max_depth=10,
      min_samples_leaf=25,
      random_state=42,
      n_jobs=-1,
  )
  rf_final.fit(X_full, y_full)

  # Package bundle
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  trained_on_date = str(pd.to_datetime(df["date"]).max().date())
  bundle = {
      "model": rf_final,
      "classes": list(rf_final.classes_),
      "latest_team_stats": latest_team_stats,
      "feature_order": feature_order,
      "trained_on_date": trained_on_date,
      "model_version": "1.0.0",
      "valid_teams": valid_teams,
      "metrics_pre_2022_trained_model": metrics_report,
      "metrics": metrics_report,
      "permutation_importance": perm_data,
  }

  with open(MODEL_PKL_PATH, "wb") as f:
    pickle.dump(bundle, f)

  print(f"\nArtifact bundle saved successfully to: {MODEL_PKL_PATH}")
  print(f"File size: {os.path.getsize(MODEL_PKL_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
  train_and_evaluate()
