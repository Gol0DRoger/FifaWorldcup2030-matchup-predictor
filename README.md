# FIFA World Cup 2030 Matchup Predictor 🏆

A professional machine learning serving project that predicts FIFA World Cup 2030 matchup outcomes using historical international match results, custom Elo ratings, recent form, and rolling goal metrics.

---


## Screenshots
<img width="959" height="539" alt="Screenshot 2026-11-24 001646" src="https://github.com/user-attachments/assets/6ed429a0-2d27-48bf-8d71-5571959d2f30" />
<img width="959" height="539" alt="Screenshot 2026-11-24 001936" src="https://github.com/user-attachments/assets/26186031-4d44-45ee-8255-fb7c50ebd31a" />
<img width="947" height="471" alt="Screenshot 2026-11-24 001453" src="https://github.com/user-attachments/assets/d3785a38-4e32-49b2-a6c3-9758f2df4146" />

---

## Architecture & Project Structure

```
Fifa2030Match/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application with lifespan management & thin routes
│   ├── schemas.py       # Pydantic v2 request & response schemas
│   └── service.py       # Inference logic, dynamic class mapping, neutral-venue symmetry
├── artifacts/
│   └── model.pkl        # Pickle bundle (model, latest team stats, feature order, metadata)
├── data/
│   └── results.csv      # Kaggle international football results (1872 - 2026)
├── frontend/
│   └── streamlit_app.py # Streamlit UI with team dropdowns, venue selection & cards
├── train.py             # Feature engineering, Elo computation, baseline benchmarking
├── verify_server.py     # End-to-end integration test & sanity verification script
├── requirements.txt     # Pinned dependencies (scikit-learn==1.9.1)
└── README.md
```

---

## 1. How to Run

### Setup Environment

Ensure Python 3.11+ is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

### (Optional) Retrain Model

To regenerate the feature pipeline, evaluate models against baselines, and re-export `artifacts/model.pkl`:

```bash
python train.py
```

### Start the FastAPI Backend

Run the API server with Uvicorn:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

* API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
* Health Check: `http://127.0.0.1:8000/health`
* Teams List: `http://127.0.0.1:8000/teams`

### Start the Streamlit Frontend

In a separate terminal window:

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Run Automated Sanity & Server Verification

To automatically start the backend, execute all sanity checks, and cleanly stop the server:

```bash
python verify_server.py
```

---

## 2. Model & Baseline Evaluation Table

Models were trained strictly on international matches from **1990 to 2021** (27,733 matches). Elo ratings were initialized in **1872** to ensure complete warmup prior to training. The models were evaluated on the held-out test split (**2022 to 2026**, 5,247 matches) as well as the **2022 FIFA World Cup** (64 matches) and **2026 FIFA World Cup** (104 matches).

| Dataset Split | Model | Accuracy | Multi-class Log Loss | vs Home Baseline (Acc) |
| :--- | :--- | :---: | :---: | :---: |
| **Test Set (2022–2026)**<br>*(N = 5,247)* | 1. Always Home Win (Majority Prior) | 47.95% | 1.0493 | (reference) |
| | 2. Elo Logistic Regression | 60.28% | 0.8754 | +12.33% |
| | **3. Random Forest (Multi-feature)** | **60.55%** | **0.8734** | **+12.60%** |
| **Holdout: 2022 World Cup**<br>*(N = 64)* | 1. Always Home Win (Majority Prior) | 43.75% | 1.0741 | (reference) |
| | 2. Elo Logistic Regression | 48.44% | 1.0794 | +4.69% |
| | **3. Random Forest (Multi-feature)** | **50.00%** | **1.0466** | **+6.25%** |
| **Holdout: 2026 World Cup**<br>*(N = 104)* | 1. Always Home Win (Majority Prior) | 47.12% | 1.0547 | (reference) |
| | 2. Elo Logistic Regression | 64.42% | 0.8465 | +17.31% |
| | **3. Random Forest (Multi-feature)** | **65.38%** | **0.8439** | **+18.27%** |

*Note on Evaluation & Production Refitting:*
- The **Random Forest** outperforms both the naive home-win baseline and Elo-only logistic regression on both **accuracy** and **log loss** across all three evaluation splits.
- Benchmark metrics are preserved in `artifacts/model.pkl` under the key `metrics_pre_2022_trained_model`.
- After benchmarking and permutation importance, the final production Random Forest model is refit on **ALL matches from 1990 through the latest date** (32,980 matches) with the exact same hyperparameters and saved in the artifact bundle.

---

## 3. Permutation Feature Importance

Permutation importance measured on the test set ($N=5,247$, 5 repeats, mean drop in accuracy):

| Rank | Feature | Mean Drop in Accuracy | Std Dev | Description |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `effective_elo_diff` | **0.06240** | 0.00616 | Pre-match Elo difference adjusted for home advantage |
| 2 | `elo_diff` | **0.01894** | 0.00141 | Raw pre-match Elo rating difference |
| 3 | `elo_away` | **0.00225** | 0.00221 | Away team pre-match Elo rating |
| 4 | `neutral` | **0.00072** | 0.00056 | Binary flag (1 if match at neutral ground, else 0) |
| 5 | `elo_home` | **0.00061** | 0.00145 | Home team pre-match Elo rating |
| 6 | `rolling_gd_diff` | **0.00046** | 0.00067 | Difference in rolling 10-match goal difference |
| 7 | `form_diff` | **0.00034** | 0.00056 | Difference in points-per-match over last 5 matches |
| 8 | `home_rolling_ga` | **0.00027** | 0.00061 | Home team average goals conceded in last 10 matches |
| 9 | `away_form` | **0.00019** | 0.00048 | Away team recent points per match |
| 10 | `elo_trend_diff` | **0.00019** | 0.00045 | Difference in 5-match Elo trajectory |

---

## 4. Inference Rules & Symmetrical Invariance

1. **Tournament Scope**: Always locked internally to FIFA World Cup (`tournament_weight = 60.0`).
2. **Neutral Venue Invariance**:
   For neutral venues, matches are evaluated in both permutations (`team_a` as home, `team_b` as away, then swapped) and averaged:
   $$\text{Win}(A) = \frac{P_{A,B}(\text{home\_win}) + P_{B,A}(\text{away\_win})}{2}$$
   $$\text{Draw} = \frac{P_{A,B}(\text{draw}) + P_{B,A}(\text{draw})}{2}$$
   $$\text{Win}(B) = \frac{P_{A,B}(\text{away\_win}) + P_{B,A}(\text{home\_win})}{2}$$
   This guarantees that $\text{predict}(A, B) \equiv \text{predict}(B, A)$ with zero artificial home bias.
3. **Dynamic Class Mapping**: Class indices are mapped dynamically via `model.classes_` rather than hardcoding array offsets.
4. **Team Validation**: Exactly 211 active FIFA national associations are supported (must have contested a World Cup or qualification match and been active since 2022). Non-FIFA sides (e.g., Padania, Sealand, Guadeloupe) are rejected with HTTP 404.

---

## 5. Sanity Checks Verification

Verified via `verify_server.py`:
- **Brazil vs Weak Team**: Brazil vs San Marino yields **97.17% Brazil Win** probability (`confidence = 0.9717`, `prediction = 'team_a_win'`).
- **Neutral Venue Symmetry**:
  - `predict("Brazil", "France", "neutral")`: Brazil Win = 28.70%, Draw = 29.34%, France Win = 41.96%
  - `predict("France", "Brazil", "neutral")`: France Win = 41.96%, Draw = 29.34%, Brazil Win = 28.70%
- **Home Advantage Effect (Spain vs Portugal)**:
  - Neutral Ground: Spain Win = **63.8%**, Portugal Win = **12.9%**
  - Spain Home: Spain Win increases to **68.7%**
  - Portugal Home: Portugal Win increases to **21.5%**

---

## 6. Honest Limitations

1. **Frozen Snapshot**:
   Team ratings, rolling goals, and form are captured as of November 2026 (end of the results dataset). Real international teams experience generational transitions, tactical alterations, and injuries.
2. **Retrain Required Before 2030**:
   For actual usage during the 2030 centenary tournament, the pipeline should be re-executed with matches played between 2026 and 2030.
3. **Rarely Predicts Draws as Argmax**:
   Like virtually all probabilistic soccer models with an argmax decision rule, the model rarely outputs `"draw"` as the single most probable class. Draw probabilities typically range between 20% and 30%, which is realistic for soccer, but win probabilities for one team or the other usually exceed this threshold.
4. **No Player, Tactical, or Betting Odds Data**:
   The feature space relies strictly on aggregate historical match outcomes, Elo, and scorelines. It does not incorporate player lineups, tactical formations, expected goals (xG), or market betting odds.
