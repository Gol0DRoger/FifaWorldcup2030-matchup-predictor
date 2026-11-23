"""FastAPI application for FIFA World Cup 2030 Matchup Prediction.

Thin routes with lifespan model initialization and structured error handling.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse, PredictRequest, PredictResponse, TeamsResponse
from app.service import InvalidMatchupError, PredictorService, UnknownTeamError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service singleton
service = PredictorService()


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Loads ML model and team stats bundle once at startup."""
  logger.info("Initializing PredictorService and loading model bundle...")
  service.load_model()
  logger.info(
      "Model v%s successfully loaded (trained on %s, %d teams).",
      service.model_version,
      service.trained_on_date,
      len(service.valid_teams),
  )
  yield
  logger.info("PredictorService shutting down.")


app = FastAPI(
    title="FIFA World Cup 2030 Matchup Predictor",
    description=(
        "ML-powered match simulator predicting 2030 World Cup outcomes using"
        " historical Elo ratings, form, and rolling metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(UnknownTeamError)
async def unknown_team_handler(request: Request, exc: UnknownTeamError):
  return JSONResponse(
      status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)}
  )


@app.exception_handler(InvalidMatchupError)
async def invalid_matchup_handler(request: Request, exc: InvalidMatchupError):
  return JSONResponse(
      status_code=422,
      content={"detail": str(exc)},
  )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
  logger.error("Unhandled error: %s", exc, exc_info=True)
  return JSONResponse(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      content={"detail": "An internal server error occurred."},
  )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health():
  """Returns API health status, loaded model version, and dataset timestamp."""
  return service.get_health()


@app.get("/teams", response_model=TeamsResponse, tags=["Teams"])
async def get_teams():
  """Returns the list of active FIFA national teams available for simulation."""
  teams = service.get_teams()
  return {"teams": teams, "total_teams": len(teams)}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_match(request: PredictRequest):
  """Predicts match probabilities and outcome between two FIFA teams for World Cup 2030."""
  try:
    result = service.predict(
        team_a=request.team_a, team_b=request.team_b, venue=request.venue
    )
    return result
  except (UnknownTeamError, InvalidMatchupError):
    raise
  except Exception as e:
    logger.error("Prediction error: %s", e, exc_info=True)
    raise HTTPException(
        status_code=500, detail="Internal error during match prediction."
    )
