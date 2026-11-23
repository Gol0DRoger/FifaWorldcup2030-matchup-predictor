"""Comprehensive automated tests and sanity checks for FIFA World Cup 2030 Matchup Predictor."""

from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
  with TestClient(app) as client:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"
    assert "trained_on_date" in data
    print("✓ /health passed:", data)


def test_teams_endpoint():
  with TestClient(app) as client:
    response = client.get("/teams")
    assert response.status_code == 200
    data = response.json()
    teams = data["teams"]
    assert len(teams) == 211
    assert "Brazil" in teams
    assert "Spain" in teams
    assert "France" in teams
    assert "Argentina" in teams
    # Exclude non-FIFA / defunct sides
    assert "Padania" not in teams
    assert "Sealand" not in teams
    assert "Guadeloupe" not in teams
    assert "Yugoslavia" not in teams
    print(f"✓ /teams passed: {len(teams)} valid national teams verified.")


def test_sanity_checks():
  with TestClient(app) as client:
    # 1. Sanity Check: Brazil vs weak team (e.g. San Marino)
    res_weak = client.post(
        "/predict",
        json={"team_a": "Brazil", "team_b": "San Marino", "venue": "neutral"},
    )
    assert res_weak.status_code == 200
    data_weak = res_weak.json()
    print("Sanity Check 1 - Brazil vs San Marino (neutral):", data_weak)
    assert (
        data_weak["team_a_win"] > 0.80
    ), f"Expected high Brazil win probability, got {data_weak['team_a_win']}"
    assert data_weak["prediction"] == "team_a_win"
    print(
        f"✓ Sanity Check 1 Passed: Brazil win prob ="
        f" {data_weak['team_a_win'] * 100:.1f}%"
    )

    # 2. Sanity Check: Swapping team order on neutral gives mirrored results
    res_ab = client.post(
        "/predict",
        json={"team_a": "Brazil", "team_b": "France", "venue": "neutral"},
    )
    res_ba = client.post(
        "/predict",
        json={"team_a": "France", "team_b": "Brazil", "venue": "neutral"},
    )
    assert res_ab.status_code == 200
    assert res_ba.status_code == 200
    d_ab = res_ab.json()
    d_ba = res_ba.json()
    print("Sanity Check 2 - Brazil vs France:", d_ab)
    print("Sanity Check 2 - France vs Brazil:", d_ba)

    assert abs(d_ab["team_a_win"] - d_ba["team_b_win"]) < 1e-4
    assert abs(d_ab["team_b_win"] - d_ba["team_a_win"]) < 1e-4
    assert abs(d_ab["draw"] - d_ba["draw"]) < 1e-4
    print("✓ Sanity Check 2 Passed: Neutral order symmetry is exact.")

    # 3. Sanity Check: Home venue raises that team's win probability
    res_neutral = client.post(
        "/predict",
        json={"team_a": "Spain", "team_b": "Portugal", "venue": "neutral"},
    )
    res_spain_home = client.post(
        "/predict",
        json={"team_a": "Spain", "team_b": "Portugal", "venue": "team_a_home"},
    )
    res_portugal_home = client.post(
        "/predict",
        json={"team_a": "Spain", "team_b": "Portugal", "venue": "team_b_home"},
    )

    assert res_neutral.status_code == 200
    assert res_spain_home.status_code == 200
    assert res_portugal_home.status_code == 200

    d_neu = res_neutral.json()
    d_sh = res_spain_home.json()
    d_ph = res_portugal_home.json()

    print(
        f"Spain vs Portugal | Neutral: Spain={d_neu['team_a_win'] * 100:.1f}%,"
        f" Portugal={d_neu['team_b_win'] * 100:.1f}%"
    )
    print(
        "Spain vs Portugal | Spain Home:"
        f" Spain={d_sh['team_a_win'] * 100:.1f}%,"
        f" Portugal={d_sh['team_b_win'] * 100:.1f}%"
    )
    print(
        "Spain vs Portugal | Portugal Home:"
        f" Spain={d_ph['team_a_win'] * 100:.1f}%,"
        f" Portugal={d_ph['team_b_win'] * 100:.1f}%"
    )

    assert (
        d_sh["team_a_win"] > d_neu["team_a_win"]
    ), "Spain home win prob should exceed neutral"
    assert (
        d_ph["team_b_win"] > d_neu["team_b_win"]
    ), "Portugal home win prob should exceed neutral"
    print("✓ Sanity Check 3 Passed: Home advantage increases win probability.")


def test_error_handling():
  with TestClient(app) as client:
    # Unknown team
    res_unknown = client.post(
        "/predict",
        json={"team_a": "Padania", "team_b": "Brazil", "venue": "neutral"},
    )
    assert res_unknown.status_code == 404
    print("✓ 404 Unknown team handled correctly:", res_unknown.json())

    # Same team matchup
    res_same = client.post(
        "/predict",
        json={"team_a": "France", "team_b": "France", "venue": "neutral"},
    )
    assert res_same.status_code == 422
    print("✓ 422 Same team matchup rejected correctly:", res_same.json())

    # Invalid venue
    res_venue = client.post(
        "/predict",
        json={"team_a": "France", "team_b": "Brazil", "venue": "mars"},
    )
    assert res_venue.status_code == 422
    print("✓ 422 Invalid venue rejected correctly:", res_venue.json())


if __name__ == "__main__":
  test_health_endpoint()
  test_teams_endpoint()
  test_sanity_checks()
  test_error_handling()
  print("\n🎉 ALL TESTS AND SANITY CHECKS PASSED PERFECTLY!")
