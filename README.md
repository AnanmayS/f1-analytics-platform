# F1 Analytics Platform

A production-style local Formula 1 analytics MVP built with real historical FastF1 data, a FastAPI backend, an XGBoost modeling pipeline, and a React replay UI.

The platform can ingest real sessions, cache processed artifacts, render a deterministic 2D replay from FastF1 position data, expose telemetry and leaderboard APIs, and serve a saved XGBoost final-position predictor in the browser.

## How To Run

```bash
cd f1-analytics-platform
docker compose up --build
```

Open:

- Frontend: http://localhost:5173
- Backend health: http://localhost:8000/health
- API docs: http://localhost:8000/docs

First demo flow:

```bash
docker compose logs -f backend
```

Docker Compose does not start the full recent-season cache automatically because the first FastF1 download can take a long time. Open the frontend, ingest one race for a quick replay demo, or click `Cache recent seasons` when you want the larger model/replay dataset.

Manual data commands:

```bash
docker compose exec backend python scripts/bootstrap_data.py
docker compose exec backend python scripts/preprocess_sessions.py --seasons 2024 --events "Monaco Grand Prix" --session R --workers 1
```

Caching policy: prefer single-race ingest while exploring, then cache recent completed seasons when you want a stronger replay/model demo. Caching every race from 2018 onward is supported through bulk ingest, but it is slower, larger on disk, and unnecessary for normal local development.

## Architecture

```text
f1-analytics-platform/
  backend/
    app/
      api/          FastAPI routers by domain
      core/         settings, logging, session normalization
      db/           SQLAlchemy session and SQLite bootstrap
      models/       metadata tables for ingested sessions
      schemas/      Pydantic response/request contracts
      services/     FastF1, preprocessing, artifact access, ML services
    scripts/        reusable CLI pipeline commands
    tests/          focused backend tests
    data/           local cache, processed artifacts, model files
  frontend/
    src/
      api/          typed API client
      components/   replay, charts, leaderboards, predictions
      pages/        home, replay, model evaluation
  docker-compose.yml
```

## Backend API

Implemented endpoints:

- `GET /health`
- `GET /api/seasons`
- `GET /api/events?season=2024`
- `GET /api/sessions?season=2024&event=Monaco%20Grand%20Prix`
- `POST /api/ingest/session`
- `POST /api/ingest/bulk`
- `GET /api/ingest/bootstrap-status`
- `POST /api/ingest/bootstrap`
- `GET /api/session/summary?season=2024&event=Monaco%20Grand%20Prix&session=R`
- `GET /api/session/drivers?season=2024&event=Monaco%20Grand%20Prix&session=R`
- `GET /api/session/leaderboard?season=2024&event=Monaco%20Grand%20Prix&session=R`
- `GET /api/session/telemetry?season=2024&event=Monaco%20Grand%20Prix&session=R&driver=VER&lap=fastest`
- `GET /api/session/replay?season=2024&event=Monaco%20Grand%20Prix&session=R`
- `POST /api/model/train`
- `GET /api/model/metrics`
- `GET /api/model/feature-importance`
- `GET /api/model/predict-race?season=2026&event=Miami%20Grand%20Prix&session=R`

## Data Pipeline

FastF1 integration is centralized in `backend/app/services/fastf1_service.py`.

Every FastF1 call enables local cache under `backend/data/cache`, which is mounted into Docker so repeated runs are much faster. Processed session artifacts are written under:

```text
backend/data/processed/{season}/{event}/{session}/
```

Each ingested session writes:

- `metadata.json`
- `results.parquet` or `results.csv`
- `laps.parquet` or `laps.csv`
- `weather.parquet` or `weather.csv`
- `replay_positions.parquet` or `replay_positions.csv`
- `driver_telemetry.parquet` or `driver_telemetry.csv`
- `leaderboard.json`

Batch ingestion uses multiprocessing in `scripts/preprocess_sessions.py` and `POST /api/ingest/bulk`. Broken sessions are logged, returned as failed rows, and do not stop the whole batch.

The Docker setup can bootstrap the last four completed race seasons on demand from the home page or with `scripts/bootstrap_data.py`. A marker file at `backend/data/bootstrap_recent_races.json` prevents repeat full-season downloads after the first completed bootstrap.

For faster single-race ingest, `F1_TELEMETRY_INGEST_MODE=fastest_lap` stores one useful telemetry profile per driver instead of collecting every telemetry sample from every racing lap. Set it to `full` when you want heavier full-race telemetry artifacts, or `none` when you only need results/laps/replay positions. `F1_REPLAY_POSITION_DRIVER_LIMIT=4` also limits expensive position-trace extraction to a few drivers for track-map construction; the replay animation itself is driven by lap timing and projected onto that track.

## Modeling Approach

The prediction model uses an XGBoost sklearn-style regressor to predict each driver's final finishing position, scales numeric race features, then sorts drivers into a full predicted final grid.

Prediction endpoints intentionally reject completed historical races. Completed races are used for replay,
analysis, training, and evaluation; prediction cards are for future races only.
For 2026 future races, the predictor uses the confirmed 22-driver grid, including Cadillac, Audi,
Racing Bulls, and rookie metadata, then blends current-lineup priors with local FastF1 historical artifacts.
The API still derives a simple movement class for display:

- `1`: predicted to gain positions
- `0`: predicted to finish roughly where they start
- `-1`: predicted to lose positions

`probability_gain` is a deterministic display score derived from the predicted grid-to-finish delta, not a classifier probability.

Feature rows are driver-race granularity. Current features include:

- grid position
- field-relative grid percentile
- lap-time pace aggregates
- sector pace aggregates
- pace consistency
- speed consistency
- pit stop and tire-age signals
- driver prior race count
- driver historical average position delta
- driver historical gain rate
- driver historical average grid and finish position
- team historical average position delta
- team historical gain rate
- team historical average grid and finish position
- driver/team composite form scores
- current-lineup metadata such as rookie status and team changes
- race context such as street/high-speed profile, overtaking score, safety-car tendency, and track position importance
- weather means and rainfall proxy

For completed historical training rows, lap/sector/tire/pace features come from real FastF1 race artifacts.
For future races, those same columns are filled from the latest completed race and rolling history for the
current field because future lap telemetry does not exist yet. For future events where a confirmed grid is not
available, the predictor estimates a race-specific starting grid from current driver form, team strength, rookie
status, and circuit profile. This is a deterministic pre-race approximation, not leaked qualifying data.

The predictor loads a saved model artifact when available. If the artifact is missing and enough processed race data exists, the prediction service automatically trains the model before scoring, so the frontend does not need a manual training button. The optional `scripts/train_model.py` command still exists for offline experiments. By default it trains on the latest three processed race seasons.

The training split is grouped by race using `GroupShuffleSplit`, or held out by season when `--test-season` is provided. This avoids putting rows from the same race in both train and test sets.

The model uses recency-aware sample weights: newer seasons matter more, the latest processed season receives the largest multiplier, and in-season driver/team championship points add a small form boost. That keeps the model from over-trusting older F1 seasons after major team, regulation, or driver lineup changes.

Model artifacts are stored in `backend/data/models`:

- `position_change_model.joblib`
- `metrics.json`
- `feature_importance.json`
- `training_features.csv`

## Replay And Telemetry Notes

The replay uses real FastF1 X/Y position samples when available. Samples are bucketed into two-second replay frames so all drivers can be animated together deterministically in the browser. Each frame carries its own lap number and live leaderboard so the UI does not have to guess state from the scrubber time.

Known approximation: FastF1 historical timing, car telemetry, and position streams are not always perfectly aligned across drivers and sessions. The replay projects cars onto a reference track polyline and uses FastF1 lap position data for stable running order, but it is still a best-effort deterministic reconstruction rather than exact broadcast synchronization. The UI displays this limitation on the replay page.

The replay drawer follows the pit-wall style from [`IAmTomShaw/f1-race-replay`](https://github.com/IAmTomShaw/f1-race-replay): it shows the selected driver's current replay state, vertical throttle/brake meters, nearby cars, tire compound, stint, and recent lap chips instead of plotting the full raw telemetry stream in the main HUD.

Driver telemetry uses race replay channels and FastF1 car data where available:

- speed
- throttle
- brake
- RPM
- lap times
- stint and tire compound information

If a channel is unavailable for a session, the backend returns nulls instead of fabricating data.

## Local Development

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Tests:

```bash
cd backend
pytest
```

Useful pipeline commands:

```bash
python scripts/collect_sessions.py --seasons 2024
python scripts/preprocess_sessions.py --seasons 2024 --events "Monaco Grand Prix" --session R --workers 1
python scripts/preprocess_sessions.py --seasons 2023 2024 --session R --workers 2
python scripts/build_features.py --seasons 2023 2024
python scripts/bootstrap_data.py
python scripts/train_model.py
python scripts/train_model.py --seasons 2023 2024 --test-season 2024
python scripts/evaluate_model.py
```

If prediction fails with an XGBoost/scikit-learn compatibility error such as
`'super' object has no attribute '__sklearn_tags__'`, rebuild the backend image
or reinstall backend requirements so the compatible pinned versions are used:

```bash
docker compose build --no-cache backend
docker compose up
```

## Resume Bullets

- Built a Dockerized full-stack Formula 1 analytics platform using FastF1, FastAPI, React, and SQLite to ingest, cache, and serve real historical race telemetry, lap, weather, and result data.
- Engineered a driver-race feature pipeline and trained a saved XGBoost final-position regressor with grouped race-level evaluation artifacts, recency-weighted seasons, current-lineup handling, and realistic final-grid constraints.
- Developed an interactive replay interface with smooth 2D track animation, synced lap leaderboard, pit-wall style driver telemetry HUD, and future-race prediction cards for a resume-ready motorsport ML product demo.
