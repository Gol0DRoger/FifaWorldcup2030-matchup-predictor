"""End-to-end integration verification:

Starts Uvicorn server in subprocess, probes all HTTP endpoints via live network requests,
validates sanity checks, and terminates server cleanly.
"""

import subprocess
import sys
import time
import requests

API_HOST = "127.0.0.1"
API_PORT = 8000
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def run_verification():
  print("=" * 70, flush=True)
  print("STARTING LIVE UVICORN SERVER ON 127.0.0.1:8000...", flush=True)
  print("=" * 70, flush=True)

  proc = subprocess.Popen(
      [
          sys.executable,
          "-m",
          "uvicorn",
          "app.main:app",
          "--host",
          API_HOST,
          "--port",
          str(API_PORT),
          "--log-level",
          "warning",
      ],
      stdout=sys.stdout,
      stderr=sys.stderr,
  )

  try:
    # Wait for server to boot
    max_retries = 20
    ready = False
    for i in range(max_retries):
      try:
        r = requests.get(f"{BASE_URL}/health", timeout=1)
        if r.status_code == 200:
          ready = True
          print(f"Server is ready after {i + 1} retries!", flush=True)
          break
      except Exception as e:
        print(f"Retry {i + 1}: {e}", flush=True)
        time.sleep(0.5)

    if not ready:
      print("Error: Server failed to start in time.")
      stderr = proc.stderr.read().decode()
      print("Uvicorn stderr:", stderr)
      sys.exit(1)

    print("\n1. TESTING GET /health...")
    r_health = requests.get(f"{BASE_URL}/health", timeout=2)
    assert (
        r_health.status_code == 200
    ), f"Expected 200, got {r_health.status_code}"
    health_data = r_health.json()
    print("Health response:", health_data)
    assert health_data["status"] == "healthy"
    assert health_data["model_version"] == "1.0.0"
    assert health_data["trained_on_date"] == "2026-11-17"
    print("✓ GET /health PASSED")

    print("\n2. TESTING GET /teams...")
    r_teams = requests.get(f"{BASE_URL}/teams", timeout=2)
    assert r_teams.status_code == 200, f"Expected 200, got {r_teams.status_code}"
    teams_data = r_teams.json()
    teams = teams_data["teams"]
    print(f"Total active FIFA teams returned: {teams_data['total_teams']}")
    assert teams_data["total_teams"] == 211
    assert "Brazil" in teams and "Spain" in teams and "Argentina" in teams
    # Verify non-FIFA sides are excluded
    assert "Padania" not in teams
    assert "Sealand" not in teams
    assert "Guadeloupe" not in teams
    print("✓ GET /teams PASSED")

    print("\n3. SANITY CHECK: Brazil vs San Marino (Neutral Venue)...")
    r_weak = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "Brazil", "team_b": "San Marino", "venue": "neutral"},
        timeout=3,
    )
    assert r_weak.status_code == 200, f"Expected 200, got {r_weak.status_code}"
    d_weak = r_weak.json()
    print("Result:", d_weak)
    assert (
        d_weak["team_a_win"] > 0.85
    ), f"Expected Brazil win prob > 85%, got {d_weak['team_a_win']}"
    assert d_weak["prediction"] == "team_a_win"
    print(
        f"✓ SANITY CHECK 1 PASSED: Brazil Win Prob = {d_weak['team_a_win'] * 100:.2f}%"
    )

    print("\n4. SANITY CHECK: Neutral Order Invariance (Brazil vs France)...")
    r_bf = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "Brazil", "team_b": "France", "venue": "neutral"},
        timeout=3,
    )
    r_fb = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "France", "team_b": "Brazil", "venue": "neutral"},
        timeout=3,
    )
    assert r_bf.status_code == 200 and r_fb.status_code == 200
    d_bf = r_bf.json()
    d_fb = r_fb.json()
    print("Brazil vs France:", d_bf)
    print("France vs Brazil:", d_fb)
    assert (
        abs(d_bf["team_a_win"] - d_fb["team_b_win"]) < 1e-4
    ), "team_a_win(A,B) != team_b_win(B,A)"
    assert (
        abs(d_bf["team_b_win"] - d_fb["team_a_win"]) < 1e-4
    ), "team_b_win(A,B) != team_a_win(B,A)"
    assert abs(d_bf["draw"] - d_fb["draw"]) < 1e-4, "draw(A,B) != draw(B,A)"
    print("✓ SANITY CHECK 2 PASSED: Predict(A,B) perfectly mirrors Predict(B,A)")

    print(
        "\n5. SANITY CHECK: Home Venue Impact on Win Probability (Spain vs"
        " Portugal)..."
    )
    r_neu = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "Spain", "team_b": "Portugal", "venue": "neutral"},
        timeout=3,
    )
    r_sp_home = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "Spain", "team_b": "Portugal", "venue": "team_a_home"},
        timeout=3,
    )
    r_pt_home = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "Spain", "team_b": "Portugal", "venue": "team_b_home"},
        timeout=3,
    )
    assert (
        r_neu.status_code == 200
        and r_sp_home.status_code == 200
        and r_pt_home.status_code == 200
    )
    d_neu = r_neu.json()
    d_sh = r_sp_home.json()
    d_ph = r_pt_home.json()
    print(
        f"Neutral Venue: Spain={d_neu['team_a_win'] * 100:.1f}%,"
        f" Portugal={d_neu['team_b_win'] * 100:.1f}%"
    )
    print(
        f"Spain Home:    Spain={d_sh['team_a_win'] * 100:.1f}%,"
        f" Portugal={d_sh['team_b_win'] * 100:.1f}%"
    )
    print(
        f"Portugal Home: Spain={d_ph['team_a_win'] * 100:.1f}%,"
        f" Portugal={d_ph['team_b_win'] * 100:.1f}%"
    )
    assert (
        d_sh["team_a_win"] > d_neu["team_a_win"]
    ), "Spain home win prob should exceed neutral"
    assert (
        d_ph["team_b_win"] > d_neu["team_b_win"]
    ), "Portugal home win prob should exceed neutral"
    print(
        "✓ SANITY CHECK 3 PASSED: Home venue raises team win probability as"
        " expected"
    )

    print("\n6. TESTING ERROR HANDLING...")
    # Unknown team -> 404
    r_unknown = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "Padania", "team_b": "Brazil", "venue": "neutral"},
        timeout=3,
    )
    print("Unknown team response status:", r_unknown.status_code)
    assert (
        r_unknown.status_code == 404
    ), f"Expected 404, got {r_unknown.status_code}"
    print("✓ 404 Unknown team error verified:", r_unknown.json())

    # Same team matchup -> 422
    r_same = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "Argentina", "team_b": "Argentina", "venue": "neutral"},
        timeout=3,
    )
    print("Same team response status:", r_same.status_code)
    assert r_same.status_code == 422, f"Expected 422, got {r_same.status_code}"
    print("✓ 422 Same team error verified:", r_same.json())

    # Invalid venue -> 422
    r_bad_venue = requests.post(
        f"{BASE_URL}/predict",
        json={"team_a": "Argentina", "team_b": "France", "venue": "moon_stadium"},
        timeout=3,
    )
    print("Bad venue response status:", r_bad_venue.status_code)
    assert (
        r_bad_venue.status_code == 422
    ), f"Expected 422, got {r_bad_venue.status_code}"
    print("✓ 422 Bad venue error verified:", r_bad_venue.json())

    print("\n" + "=" * 70)
    print("🎉 ALL ENDPOINTS AND SANITY CHECKS VERIFIED SUCCESSFULLY!")
    print("=" * 70)

  finally:
    print("\nTerminating Uvicorn test server...")
    proc.terminate()
    proc.wait(timeout=5)
    print("Server stopped cleanly.")


if __name__ == "__main__":
  run_verification()
