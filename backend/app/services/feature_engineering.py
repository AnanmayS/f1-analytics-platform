from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.current_lineup import CurrentDriver, current_lineup_for_season, team_aliases
from app.services.artifact_store import ArtifactStore


MODEL_FEATURES = [
    "season",
    "event",
    "circuit_name",
    "driver",
    "team",
    "grid_position",
    "field_grid_percentile",
    "driver_prior_races",
    "driver_avg_delta_prior",
    "driver_gain_rate_prior",
    "driver_avg_grid_prior",
    "driver_avg_finish_prior",
    "team_prior_races",
    "team_avg_delta_prior",
    "team_gain_rate_prior",
    "air_temp_mean",
    "track_temp_mean",
    "rainfall_rate",
]

PERFORMANCE_FEATURES = [
    "avg_lap_time",
    "best_lap_time",
    "worst_lap_time",
    "lap_time_std",
    "avg_sector1",
    "avg_sector2",
    "avg_sector3",
    "total_pitstops",
    "avg_tire_age",
    "avg_speed",
    "max_speed",
    "speed_consistency",
    "consistency_score",
]

RACE_CONTEXT_FEATURES = [
    "race_round",
    "track_speed_score",
    "track_overtaking_score",
    "track_degradation_score",
    "track_safety_car_score",
    "track_position_importance",
    "track_wet_probability",
    "driver_track_experience_score",
    "driver_track_avg_finish",
    "driver_track_best_finish",
    "event_variation_score",
    "estimated_grid_score",
]

FINAL_POSITION_FEATURES = [
    "grid_position",
    "field_grid_percentile",
    *PERFORMANCE_FEATURES,
    *RACE_CONTEXT_FEATURES,
    "air_temp_mean",
    "track_temp_mean",
    "rainfall_rate",
    "is_wet",
    "is_rookie",
    "championship_points_before_race",
    "position_in_championship_before_race",
    "avg_finish_last_5_races",
    "avg_qualifying_position_last_5",
    "dnfs_last_10_races",
    "best_finish_on_this_track",
    "teammate_points_difference_before_race",
    "races_completed_this_season",
    "team_points_before_race",
    "avg_finish_last_5_for_team",
    "team_best_finish_this_season",
    "team_avg_qualifying_position",
    "points_per_race",
    "championship_position_normalized",
    "qualifying_vs_race",
    "qualifying_consistency",
    "track_experience_score",
    "teammate_advantage",
    "performance_trend",
    "grid_vs_typical_qual",
    "rookie_grid_penalty",
    "rookie_experience_bonus",
    "driver_prior_races",
    "driver_avg_delta_prior",
    "driver_gain_rate_prior",
    "driver_avg_grid_prior",
    "driver_avg_finish_prior",
    "team_prior_races",
    "team_avg_delta_prior",
    "team_gain_rate_prior",
    "team_avg_grid_prior",
    "team_avg_finish_prior",
    "recent_form_score",
    "experience_level",
    "reliability_score",
    "team_strength_score",
    "driver_skill_score",
    "expected_finish_from_performance",
    "grid_advantage",
    "grid_x_recent_form",
    "grid_x_experience",
    "grid_x_reliability",
]

NUMERIC_FEATURES = [
    "season",
    "grid_position",
    "field_grid_percentile",
    "driver_prior_races",
    "driver_avg_delta_prior",
    "driver_gain_rate_prior",
    "driver_avg_grid_prior",
    "driver_avg_finish_prior",
    "team_prior_races",
    "team_avg_delta_prior",
    "team_gain_rate_prior",
    "air_temp_mean",
    "track_temp_mean",
    "rainfall_rate",
]

CATEGORICAL_FEATURES = ["event", "circuit_name", "driver", "team"]


@dataclass(frozen=True)
class FeatureBuildResult:
    frame: pd.DataFrame
    source_sessions: int


def bucket_position_delta(grid_position: int | float | None, finishing_position: int | float | None) -> int | None:
    """Classify grid-to-finish movement as gain/flat/loss.

    Positive race movement is grid position minus finishing position. A driver who
    starts P10 and finishes P7 gained three positions.
    """

    if grid_position is None or finishing_position is None:
        return None
    try:
        grid = int(float(grid_position))
        finish = int(float(finishing_position))
    except (TypeError, ValueError):
        return None
    if grid <= 0 or finish <= 0:
        return None
    delta = grid - finish
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def position_delta(grid_position: int | float | None, finishing_position: int | float | None) -> int | None:
    if grid_position is None or finishing_position is None:
        return None
    try:
        grid = int(float(grid_position))
        finish = int(float(finishing_position))
    except (TypeError, ValueError):
        return None
    if grid <= 0 or finish <= 0:
        return None
    return grid - finish


class FeatureBuilder:
    def __init__(self) -> None:
        self.store = ArtifactStore()

    def build_from_processed(self, seasons: list[int] | None = None, session_code: str = "R") -> FeatureBuildResult:
        rows: list[dict[str, Any]] = []
        sessions = 0
        for session_dir in self.store.list_session_dirs():
            metadata_path = session_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = self.store.read_json(metadata_path)
            if metadata.get("session") != session_code:
                continue
            if seasons and int(metadata.get("season")) not in seasons:
                continue
            session_rows = self._rows_for_session(session_dir, metadata)
            if session_rows:
                sessions += 1
                rows.extend(session_rows)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = self._add_historical_features(frame)
        return FeatureBuildResult(frame=frame, source_sessions=sessions)

    def build_for_session(self, season: int, event: str, session: str) -> pd.DataFrame:
        session_dir = self.store.session_dir(season, event, session)
        metadata = self.store.read_json(session_dir / "metadata.json")
        return pd.DataFrame(self._rows_for_session(session_dir, metadata, include_target=True))

    def build_future_rows(self, season: int, event: str, drivers: list[str] | None = None) -> pd.DataFrame:
        history = self.build_from_processed().frame
        if history.empty:
            return pd.DataFrame()
        history = history.sort_values(["season", "event_date", "race_id", "driver"]).copy()
        lineup = current_lineup_for_season(season)
        if lineup and not drivers:
            future = self._future_rows_from_current_lineup(history, season, event, lineup)
        else:
            if drivers:
                latest = history.groupby("driver", as_index=False).tail(1)
            else:
                latest_race_id = str(history.iloc[-1]["race_id"])
                latest = history[history["race_id"] == latest_race_id].copy()
            if drivers:
                latest = latest[latest["driver"].isin(drivers)]
            future = latest.copy()
        future["season"] = season
        future["event"] = event
        future["session"] = "R"
        future["race_id"] = f"{season}::{event}"
        future["circuit_name"] = event
        future["finishing_position"] = None
        future["actual_position_delta"] = None
        future["target"] = None
        future = self._refresh_future_history_features(history, future)
        future = self._add_track_history_features(history, future, event)
        future = self._estimate_future_grid(future, event)
        return future

    def _rows_for_session(self, session_dir: Path, metadata: dict[str, Any], include_target: bool = True) -> list[dict[str, Any]]:
        try:
            results = self.store.read_frame(session_dir / "results")
            laps = self.store.read_frame(session_dir / "laps")
            weather = self.store.read_frame(session_dir / "weather")
        except FileNotFoundError:
            return []
        if results.empty or laps.empty:
            return []
        lap_positions = self._lap_positions(laps)
        performance_features = self._race_performance_features(laps)
        weather_features = self._weather_features(weather)
        race_context = self._race_context_features(str(metadata.get("event") or ""), int(metadata.get("round_number") or 0))

        rows: list[dict[str, Any]] = []
        for _, result in results.iterrows():
            driver = result.get("Abbreviation") or result.get("DriverId")
            if not driver:
                continue
            driver_positions = lap_positions.get(str(driver), {})
            grid = _to_int(result.get("GridPosition")) or driver_positions.get("first_position")
            finish = _to_int(result.get("Position")) or driver_positions.get("last_position")
            target = bucket_position_delta(grid, finish)
            actual_delta = position_delta(grid, finish)
            if include_target and target is None:
                continue

            row = {
                "season": int(metadata.get("season")),
                "event": metadata.get("event"),
                "session": metadata.get("session"),
                "race_id": f"{metadata.get('season')}::{metadata.get('event')}",
                "event_date": metadata.get("event_date"),
                "circuit_name": metadata.get("circuit_name") or metadata.get("event"),
                "driver": str(driver),
                "team": result.get("TeamName"),
                "grid_position": grid,
                "finishing_position": finish,
                "actual_position_delta": actual_delta,
                "points": _to_float(result.get("Points")) or 0.0,
                "status": result.get("Status"),
                "event_variation_score": _driver_event_variation(str(driver), str(metadata.get("event") or "")),
                "estimated_grid_score": 1.0 / max(float(grid or 20), 1.0),
                **race_context,
                **weather_features,
                **performance_features.get(str(driver), {}),
            }
            if include_target:
                row["target"] = target
            rows.append(row)
        return rows

    def _lap_positions(self, laps: pd.DataFrame) -> dict[str, dict[str, int | None]]:
        if laps.empty or "Driver" not in laps or "Position" not in laps or "LapNumber" not in laps:
            return {}
        frame = laps.copy()
        frame["LapNumber"] = pd.to_numeric(frame["LapNumber"], errors="coerce")
        frame["Position"] = pd.to_numeric(frame["Position"], errors="coerce")
        lookup: dict[str, dict[str, int | None]] = {}
        for driver, group in frame.dropna(subset=["LapNumber", "Position"]).groupby("Driver"):
            ordered = group.sort_values("LapNumber")
            lookup[str(driver)] = {
                "first_position": _to_int(ordered.iloc[0].get("Position")),
                "last_position": _to_int(ordered.iloc[-1].get("Position")),
            }
        return lookup

    def _add_historical_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["_event_date_sort"] = pd.to_datetime(frame.get("event_date"), errors="coerce", utc=True)
        frame = frame.sort_values(["season", "_event_date_sort", "event", "driver"]).reset_index(drop=True)
        frame["field_grid_percentile"] = frame.groupby("race_id")["grid_position"].rank(pct=True)
        gain_indicator = (frame["target"] == 1).astype(float)

        driver_groups = frame.groupby("driver", sort=False)
        team_groups = frame.groupby("team", sort=False)
        frame["driver_prior_races"] = driver_groups.cumcount()
        frame["driver_avg_delta_prior"] = driver_groups["actual_position_delta"].transform(lambda series: series.shift().expanding().mean())
        frame["driver_gain_rate_prior"] = gain_indicator.groupby(frame["driver"]).transform(lambda series: series.shift().expanding().mean())
        frame["driver_avg_grid_prior"] = driver_groups["grid_position"].transform(lambda series: series.shift().expanding().mean())
        frame["driver_avg_finish_prior"] = driver_groups["finishing_position"].transform(lambda series: series.shift().expanding().mean())
        frame["team_prior_races"] = team_groups.cumcount()
        frame["team_avg_delta_prior"] = team_groups["actual_position_delta"].transform(lambda series: series.shift().expanding().mean())
        frame["team_gain_rate_prior"] = gain_indicator.groupby(frame["team"]).transform(lambda series: series.shift().expanding().mean())
        frame["team_avg_grid_prior"] = team_groups["grid_position"].transform(lambda series: series.shift().expanding().mean())
        frame["team_avg_finish_prior"] = team_groups["finishing_position"].transform(lambda series: series.shift().expanding().mean())
        frame["races_completed_this_season"] = frame.groupby(["season", "driver"], sort=False).cumcount()
        frame["championship_points_before_race"] = frame.groupby(["season", "driver"], sort=False)["points"].transform(
            lambda series: series.shift().fillna(0).cumsum()
        )
        frame["team_points_before_race"] = frame.groupby(["season", "team"], sort=False)["points"].transform(
            lambda series: series.shift().fillna(0).cumsum()
        )
        frame["avg_finish_last_5_races"] = driver_groups["finishing_position"].transform(lambda series: series.shift().rolling(5, min_periods=1).mean())
        frame["avg_qualifying_position_last_5"] = driver_groups["grid_position"].transform(lambda series: series.shift().rolling(5, min_periods=1).mean())
        status_text = frame.get("status", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
        dnf_indicator = (~status_text.str.contains("finished|\\+", regex=True) & (status_text != "")).astype(float)
        frame["dnfs_last_10_races"] = dnf_indicator.groupby(frame["driver"]).transform(lambda series: series.shift().rolling(10, min_periods=1).sum())
        frame["best_finish_on_this_track"] = frame.groupby(["driver", "circuit_name"], sort=False)["finishing_position"].transform(
            lambda series: series.shift().expanding().min()
        )
        frame["team_best_finish_this_season"] = frame.groupby(["season", "team"], sort=False)["finishing_position"].transform(
            lambda series: series.shift().expanding().min()
        )
        frame["avg_finish_last_5_for_team"] = team_groups["finishing_position"].transform(lambda series: series.shift().rolling(5, min_periods=1).mean())
        frame["team_avg_qualifying_position"] = team_groups["grid_position"].transform(lambda series: series.shift().expanding().mean())
        frame["teammate_points_difference_before_race"] = self._teammate_points_delta(frame)
        frame["position_in_championship_before_race"] = frame.groupby("race_id")["championship_points_before_race"].rank(
            ascending=False,
            method="min",
        )
        # Processed local history may start mid-career, so "no prior rows in this
        # cache" is not enough to call someone a rookie. Future current-lineup
        # rows carry explicit rookie metadata instead.
        frame["is_rookie"] = 0.0
        frame = self._add_track_history_features(frame, frame, None)
        frame = self._add_performance_scores(frame)
        return frame.drop(columns=["_event_date_sort"])

    def _weather_features(self, weather: pd.DataFrame) -> dict[str, float | None]:
        if weather.empty:
            return {"air_temp_mean": None, "track_temp_mean": None, "rainfall_rate": None, "is_wet": 0.0}
        return {
            "air_temp_mean": _series_mean(weather, "AirTemp"),
            "track_temp_mean": _series_mean(weather, "TrackTemp"),
            "rainfall_rate": _series_mean(weather, "Rainfall"),
            "is_wet": 1.0 if (_series_mean(weather, "Rainfall") or 0.0) > 0 else 0.0,
        }

    def _race_performance_features(self, laps: pd.DataFrame) -> dict[str, dict[str, float | None]]:
        """Create race-level performance features from FastF1 lap artifacts.

        This collapses noisy lap-level data into one driver-race row with pace,
        consistency, tire, pit, and speed signals. For future races, these same
        columns are populated from each driver's latest completed race history.
        """

        if laps.empty or "Driver" not in laps:
            return {}
        frame = laps.copy()
        numeric_columns = [
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "Stint",
            "TyreLife",
            "SpeedI1",
            "SpeedI2",
            "SpeedFL",
            "SpeedST",
        ]
        for column in numeric_columns:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        lookup: dict[str, dict[str, float | None]] = {}
        for driver, group in frame.groupby("Driver"):
            lap_times = group["LapTime"].dropna() if "LapTime" in group else pd.Series(dtype=float)
            speed_values = (
                group[[column for column in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"] if column in group]]
                .stack()
                .dropna()
                if any(column in group for column in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"])
                else pd.Series(dtype=float)
            )
            lap_std = _to_float(lap_times.std()) if not lap_times.empty else None
            speed_std = _to_float(speed_values.std()) if not speed_values.empty else None
            lookup[str(driver)] = {
                "avg_lap_time": _to_float(lap_times.mean()) if not lap_times.empty else None,
                "best_lap_time": _to_float(lap_times.min()) if not lap_times.empty else None,
                "worst_lap_time": _to_float(lap_times.max()) if not lap_times.empty else None,
                "lap_time_std": lap_std,
                "avg_sector1": _series_mean(group, "Sector1Time"),
                "avg_sector2": _series_mean(group, "Sector2Time"),
                "avg_sector3": _series_mean(group, "Sector3Time"),
                "total_pitstops": max(_to_float(group["Stint"].nunique() - 1) or 0.0, 0.0) if "Stint" in group else 0.0,
                "avg_tire_age": _series_mean(group, "TyreLife"),
                "avg_speed": _to_float(speed_values.mean()) if not speed_values.empty else None,
                "max_speed": _to_float(speed_values.max()) if not speed_values.empty else None,
                "speed_consistency": 1.0 / (1.0 + speed_std) if speed_std is not None else None,
                "consistency_score": 1.0 / (1.0 + lap_std) if lap_std is not None else None,
            }
        return lookup

    def _add_performance_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        def col(name: str, default: float) -> pd.Series:
            if name in frame:
                return pd.to_numeric(frame[name], errors="coerce").fillna(default)
            return pd.Series(default, index=frame.index, dtype=float)

        default_finish = _safe_series_median(frame["finishing_position"], 10.5) if "finishing_position" in frame else 10.5
        default_grid = _safe_series_median(frame["grid_position"], 10.5) if "grid_position" in frame else 10.5

        driver_finish = col("driver_avg_finish_prior", default_finish).clip(lower=1)
        driver_grid = col("driver_avg_grid_prior", default_grid).clip(lower=1)
        team_finish = col("team_avg_finish_prior", default_finish).clip(lower=1)

        frame["recent_form_score"] = 1.0 / (1.0 + driver_finish)
        frame["experience_level"] = (col("driver_prior_races", 0.0) / 20.0).clip(lower=0.0, upper=1.0)
        frame["reliability_score"] = 1.0 - (col("dnfs_last_10_races", 0.0) / 10.0).clip(lower=0.0, upper=0.8)
        frame["team_strength_score"] = 1.0 / (1.0 + team_finish)
        frame["points_per_race"] = col("championship_points_before_race", 0.0) / col("races_completed_this_season", 0.0).clip(lower=1)
        frame["championship_position_normalized"] = col("position_in_championship_before_race", default_finish) / 22.0
        frame["qualifying_vs_race"] = col("avg_qualifying_position_last_5", default_grid) - col("avg_finish_last_5_races", default_finish)
        frame["qualifying_consistency"] = 1.0 / (
            1.0
            + (col("avg_qualifying_position_last_5", default_grid) - col("grid_position", default_grid)).abs()
        )
        frame["track_experience_score"] = 1.0 / (1.0 + col("best_finish_on_this_track", default_finish).clip(lower=1))
        frame["teammate_advantage"] = col("teammate_points_difference_before_race", 0.0) / (
            col("championship_points_before_race", 0.0).abs() + 10.0
        )
        frame["performance_trend"] = (col("position_in_championship_before_race", default_finish) - driver_finish) / 10.0
        frame["grid_vs_typical_qual"] = (col("grid_position", default_grid) - col("avg_qualifying_position_last_5", default_grid)).abs() / 10.0
        rookie = col("is_rookie", 0.0)
        frame["rookie_grid_penalty"] = np.where(rookie > 0, (col("grid_position", default_grid) - 3).clip(lower=0), 0.0)
        frame["rookie_experience_bonus"] = np.where(rookie > 0, 0.0, col("races_completed_this_season", 0.0) * 0.1)
        frame["driver_skill_score"] = (
            frame["recent_form_score"] * 0.35
            + frame["experience_level"] * 0.20
            + frame["reliability_score"] * 0.15
            + col("driver_gain_rate_prior", 0.33) * 0.15
            + frame["team_strength_score"] * 0.10
            + frame["track_experience_score"] * 0.05
        )
        frame["expected_finish_from_performance"] = driver_finish * 0.45 + team_finish * 0.35 + driver_grid * 0.20
        frame["grid_advantage"] = frame["expected_finish_from_performance"] - col("grid_position", default_grid)
        frame["grid_x_recent_form"] = col("grid_position", default_grid) * frame["recent_form_score"]
        frame["grid_x_experience"] = col("grid_position", default_grid) * frame["experience_level"]
        frame["grid_x_reliability"] = col("grid_position", default_grid) * frame["reliability_score"]
        return frame

    def _refresh_future_history_features(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        future = future.copy()
        gain_indicator = (history["target"] == 1).astype(float) if "target" in history else pd.Series(0.0, index=history.index)
        for index, row in future.iterrows():
            driver_history = history[history["driver"] == row["driver"]]
            team_history = history[history["team"] == row["team"]]
            if team_history.empty:
                aliases = team_aliases(str(row.get("team") or ""))
                team_history = history[history["team"].isin(aliases)]
            if not driver_history.empty:
                future.at[index, "driver_prior_races"] = len(driver_history)
                future.at[index, "driver_avg_delta_prior"] = driver_history["actual_position_delta"].mean()
                future.at[index, "driver_gain_rate_prior"] = gain_indicator.loc[driver_history.index].mean()
                future.at[index, "driver_avg_grid_prior"] = driver_history["grid_position"].mean()
                future.at[index, "driver_avg_finish_prior"] = driver_history["finishing_position"].mean()
                future.at[index, "avg_finish_last_5_races"] = driver_history["finishing_position"].tail(5).mean()
                future.at[index, "avg_qualifying_position_last_5"] = driver_history["grid_position"].tail(5).mean()
                future.at[index, "races_completed_this_season"] = len(driver_history[driver_history["season"] == driver_history["season"].max()])
            if not team_history.empty:
                future.at[index, "team_prior_races"] = len(team_history)
                future.at[index, "team_avg_delta_prior"] = team_history["actual_position_delta"].mean()
                future.at[index, "team_gain_rate_prior"] = gain_indicator.loc[team_history.index].mean()
                future.at[index, "team_avg_grid_prior"] = team_history["grid_position"].mean()
                future.at[index, "team_avg_finish_prior"] = team_history["finishing_position"].mean()
                future.at[index, "avg_finish_last_5_for_team"] = team_history["finishing_position"].tail(5).mean()
                future.at[index, "team_best_finish_this_season"] = team_history["finishing_position"].min()
                future.at[index, "team_avg_qualifying_position"] = team_history["grid_position"].mean()
                future.at[index, "team_points_before_race"] = team_history.get("points", pd.Series(0.0, index=team_history.index)).sum()
        return self._add_performance_scores(future)

    def _teammate_points_delta(self, frame: pd.DataFrame) -> pd.Series:
        if "points" not in frame:
            return pd.Series(0.0, index=frame.index)
        deltas = pd.Series(0.0, index=frame.index, dtype=float)
        running: dict[tuple[int, str], dict[str, float]] = {}
        ordered = frame.sort_values(["season", "_event_date_sort", "event", "team", "driver"])
        for index, row in ordered.iterrows():
            key = (int(row.get("season") or 0), str(row.get("team") or ""))
            driver = str(row.get("driver") or "")
            team_points = running.setdefault(key, {})
            driver_points = team_points.get(driver, 0.0)
            teammate_points = sum(points for name, points in team_points.items() if name != driver)
            teammate_count = max(len([name for name in team_points if name != driver]), 1)
            deltas.at[index] = driver_points - (teammate_points / teammate_count)
            team_points[driver] = driver_points + (_to_float(row.get("points")) or 0.0)
        return deltas

    def _future_rows_from_current_lineup(
        self,
        history: pd.DataFrame,
        season: int,
        event: str,
        lineup: list[CurrentDriver],
    ) -> pd.DataFrame:
        rows: list[pd.Series] = []
        template = history.groupby("driver", as_index=False).tail(1).set_index("driver", drop=False)
        for entry in lineup:
            if entry.abbreviation in template.index:
                row = template.loc[entry.abbreviation].copy()
            else:
                row = self._surrogate_row_for_current_driver(history, entry)
            row["driver"] = entry.abbreviation
            row["team"] = entry.team
            row["full_name"] = entry.full_name
            row["driver_number"] = entry.driver_number
            row["is_rookie"] = float(entry.is_rookie)
            row["championship_points_before_race"] = float(entry.championship_points)
            row["grid_position"] = float(entry.baseline_grid)
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)

    def _surrogate_row_for_current_driver(self, history: pd.DataFrame, entry: CurrentDriver) -> pd.Series:
        aliases = team_aliases(entry.team)
        if entry.previous_team:
            aliases.extend(team_aliases(entry.previous_team))
        team_history = history[history["team"].isin(dict.fromkeys(aliases))]
        if not team_history.empty:
            row = team_history.sort_values(["season", "event_date", "race_id"]).tail(1).iloc[0].copy()
        else:
            row = history.sort_values(["season", "event_date", "race_id"]).iloc[-1].copy()
        row["driver_prior_races"] = 0 if entry.is_rookie else max(_to_float(row.get("driver_prior_races")) or 5.0, 5.0)
        row["driver_avg_grid_prior"] = entry.baseline_grid
        row["driver_avg_finish_prior"] = entry.baseline_grid + (2.0 if entry.is_rookie else 0.5)
        row["driver_gain_rate_prior"] = 0.2 if entry.is_rookie else 0.35
        row["driver_avg_delta_prior"] = -0.5 if entry.is_rookie else 0.0
        return row

    def _race_context_features(self, event: str, round_number: int | None = None) -> dict[str, float]:
        profile = _track_profile(event)
        return {
            "race_round": float(round_number or _round_estimate(event)),
            **profile,
        }

    def _add_track_history_features(self, history: pd.DataFrame, frame: pd.DataFrame, event: str | None) -> pd.DataFrame:
        frame = frame.copy()
        if event is not None:
            for key, value in self._race_context_features(event).items():
                frame[key] = value
            track_key = _event_key(event)
        else:
            track_key = None
        for index, row in frame.iterrows():
            driver_history = history[history["driver"] == row["driver"]]
            if track_key is None:
                event_history = driver_history[
                    driver_history["circuit_name"].fillna(driver_history["event"]).map(_event_key)
                    .map(lambda value: _same_track(value, _event_key(str(row.get("circuit_name") or row.get("event") or ""))))
                ]
            else:
                event_history = driver_history[
                    driver_history["circuit_name"].fillna(driver_history["event"]).map(_event_key).map(lambda value: _same_track(value, track_key))
                ]
            if event_history.empty:
                frame.at[index, "driver_track_avg_finish"] = row.get("driver_avg_finish_prior", 10.5)
                frame.at[index, "driver_track_best_finish"] = row.get("driver_avg_finish_prior", 10.5)
                frame.at[index, "driver_track_experience_score"] = 0.0
            else:
                frame.at[index, "driver_track_avg_finish"] = event_history["finishing_position"].mean()
                frame.at[index, "driver_track_best_finish"] = event_history["finishing_position"].min()
                frame.at[index, "driver_track_experience_score"] = min(len(event_history) / 5.0, 1.0)
        return frame

    def _estimate_future_grid(self, future: pd.DataFrame, event: str) -> pd.DataFrame:
        future = future.copy()
        profile = _track_profile(event)
        for column in FINAL_POSITION_FEATURES:
            if column not in future:
                future[column] = np.nan
        form = future["driver_skill_score"].fillna(0.1)
        team = future["team_strength_score"].fillna(0.08)
        track_exp = future["driver_track_experience_score"].fillna(0.0)
        qualifying = 1.0 / (1.0 + future["avg_qualifying_position_last_5"].fillna(future["grid_position"]).clip(lower=1))
        speed_fit = future["avg_speed"].fillna(future["avg_speed"].median()).rank(pct=True).fillna(0.5)
        consistency_fit = future["consistency_score"].fillna(future["consistency_score"].median()).rank(pct=True).fillna(0.5)
        rookie_penalty = future["is_rookie"].fillna(0) * (0.02 + profile["track_position_importance"] * 0.04)
        deterministic_variation = future["driver"].map(lambda driver: _driver_event_variation(str(driver), event))
        score = (
            team * (0.35 + profile["track_speed_score"] * 0.15)
            + form * 0.30
            + qualifying * 0.18
            + track_exp * profile["track_position_importance"] * 0.12
            + speed_fit * profile["track_speed_score"] * 0.10
            + consistency_fit * (1.0 - profile["track_overtaking_score"]) * 0.08
            + deterministic_variation
            - rookie_penalty
        )
        future["_estimated_grid_score"] = score
        future["grid_position"] = score.rank(ascending=False, method="first").astype(int)
        future["field_grid_percentile"] = future["grid_position"].rank(pct=True)
        future["grid_advantage"] = future["expected_finish_from_performance"].fillna(10.5) - future["grid_position"]
        future["grid_x_recent_form"] = future["grid_position"] * future["recent_form_score"].fillna(0.08)
        future["grid_x_experience"] = future["grid_position"] * future["experience_level"].fillna(0)
        future["grid_x_reliability"] = future["grid_position"] * future["reliability_score"].fillna(1)
        future["rookie_grid_penalty"] = np.where(future["is_rookie"].fillna(0) > 0, (future["grid_position"] - 3).clip(lower=0), 0.0)
        future["estimated_grid_score"] = future["_estimated_grid_score"]
        future["event_variation_score"] = future["driver"].map(lambda driver: _driver_event_variation(str(driver), event))
        return future


def _series_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    return _to_float(pd.to_numeric(frame[column], errors="coerce").mean())


def _safe_series_median(series: pd.Series, fallback: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return fallback
    return float(values.median())


def _event_key(event: str) -> str:
    return (
        str(event)
        .lower()
        .replace("grand prix", "")
        .replace("gp", "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )


def _same_track(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _round_estimate(event: str) -> int:
    order = {
        "australian": 1,
        "chinese": 2,
        "japanese": 3,
        "bahrain": 4,
        "saudi arabian": 5,
        "miami": 6,
        "emilia romagna": 7,
        "monaco": 8,
        "spanish": 9,
        "canadian": 10,
        "austrian": 11,
        "british": 12,
        "belgian": 13,
        "hungarian": 14,
        "dutch": 15,
        "italian": 16,
        "azerbaijan": 17,
        "singapore": 18,
        "united states": 19,
        "mexico city": 20,
        "sao paulo": 21,
        "las vegas": 22,
        "qatar": 23,
        "abu dhabi": 24,
    }
    key = _event_key(event)
    for name, round_number in order.items():
        if name in key:
            return round_number
    return 12


def _track_profile(event: str) -> dict[str, float]:
    key = _event_key(event)
    profiles: list[tuple[tuple[str, ...], dict[str, float]]] = [
        (("monaco", "singapore"), {"track_speed_score": 0.20, "track_overtaking_score": 0.12, "track_degradation_score": 0.35, "track_safety_car_score": 0.85, "track_position_importance": 0.95, "track_wet_probability": 0.20}),
        (("azerbaijan", "las vegas", "miami", "saudi"), {"track_speed_score": 0.88, "track_overtaking_score": 0.72, "track_degradation_score": 0.45, "track_safety_car_score": 0.65, "track_position_importance": 0.55, "track_wet_probability": 0.10}),
        (("monza", "italian", "belgian", "austrian", "qatar"), {"track_speed_score": 0.95, "track_overtaking_score": 0.70, "track_degradation_score": 0.50, "track_safety_car_score": 0.35, "track_position_importance": 0.45, "track_wet_probability": 0.18}),
        (("hungarian", "spanish", "emilia", "dutch", "japanese"), {"track_speed_score": 0.58, "track_overtaking_score": 0.35, "track_degradation_score": 0.62, "track_safety_car_score": 0.35, "track_position_importance": 0.70, "track_wet_probability": 0.22}),
        (("british", "canadian", "australian", "chinese", "bahrain"), {"track_speed_score": 0.70, "track_overtaking_score": 0.55, "track_degradation_score": 0.60, "track_safety_car_score": 0.45, "track_position_importance": 0.58, "track_wet_probability": 0.24}),
        (("mexico", "sao paulo", "abu dhabi"), {"track_speed_score": 0.62, "track_overtaking_score": 0.55, "track_degradation_score": 0.55, "track_safety_car_score": 0.42, "track_position_importance": 0.55, "track_wet_probability": 0.18}),
    ]
    for names, profile in profiles:
        if any(name in key for name in names):
            return profile.copy()
    return {
        "track_speed_score": 0.60,
        "track_overtaking_score": 0.50,
        "track_degradation_score": 0.50,
        "track_safety_car_score": 0.45,
        "track_position_importance": 0.55,
        "track_wet_probability": 0.18,
    }


def _driver_event_variation(driver: str, event: str) -> float:
    # Small deterministic nudge so estimated pre-race grids vary by circuit
    # without adding randomness or pretending future qualifying has happened.
    value = sum((index + 1) * ord(char) for index, char in enumerate(f"{driver}:{_event_key(event)}"))
    return ((value % 101) - 50) / 1200.0


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None
