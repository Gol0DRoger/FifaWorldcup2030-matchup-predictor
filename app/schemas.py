"""Pydantic schemas for the FIFA World Cup 2030 Matchup Predictor API."""

from typing import Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
  status: str = Field(..., example="healthy")
  model_version: str = Field(..., example="1.0.0")
  trained_on_date: str = Field(..., example="2026-11-17")


class TeamsResponse(BaseModel):
  teams: list[str]
  total_teams: int


class PredictRequest(BaseModel):
  team_a: str = Field(..., example="Brazil", description="Name of Team A")
  team_b: str = Field(..., example="France", description="Name of Team B")
  venue: Literal["neutral", "team_a_home", "team_b_home"] = Field(
      default="neutral",
      description=(
          "Match venue: 'neutral' (default), 'team_a_home', or 'team_b_home'"
      ),
  )


class PredictResponse(BaseModel):
  team_a: str
  team_b: str
  venue: str
  team_a_win: float = Field(
      ..., description="Probability of Team A winning (0.0 to 1.0)"
  )
  draw: float = Field(
      ..., description="Probability of a draw in 90 min (0.0 to 1.0)"
  )
  team_b_win: float = Field(
      ..., description="Probability of Team B winning (0.0 to 1.0)"
  )
  prediction: str = Field(
      ..., description="Predicted outcome: 'team_a_win', 'draw', or 'team_b_win'"
  )
  confidence: float = Field(
      ..., description="Highest class probability (uncalibrated RF output)"
  )
  model_version: str
