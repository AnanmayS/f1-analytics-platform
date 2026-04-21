from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


DEFAULT_POSITION = 10.5


@dataclass
class PredictionFrame:
    """Container for exact-position predictions before API serialization."""

    results: pd.DataFrame
    prepared_features: pd.DataFrame


class F1FinalPositionRegressor:
    """XGBoost model for predicting exact finishing positions.

    The model scales numeric performance features, fills missing values with
    training medians, and sorts each race into a constrained final grid.
    """

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.fill_values: dict[str, float] = {}
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False

    def fit(self, x: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | pd.Series | None = None) -> None:
        prepared = self._prepare_features(x, fit=True)
        scaled = self.scaler.fit_transform(prepared)
        self.model.fit(scaled, y.astype(float), sample_weight=sample_weight)
        self.is_trained = True

    def predict_raw(self, x: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        prepared = self._prepare_features(x, fit=False)
        scaled = self.scaler.transform(prepared)
        return np.asarray(self.model.predict(scaled), dtype=float)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(x)
        return np.clip(np.round(raw), 1, max(20, len(x))).astype(int)

    def predict_race(self, x: pd.DataFrame, apply_constraints: bool = True) -> PredictionFrame:
        prepared = self._prepare_features(x, fit=False)
        raw_predictions = self.predict_raw(prepared)
        rounded = np.clip(np.round(raw_predictions), 1, max(20, len(prepared))).astype(int)
        has_minimal_data = self._has_minimal_data(prepared)

        if apply_constraints and has_minimal_data:
            rounded = self._apply_grid_position_anchor(rounded.astype(float), prepared)
        if apply_constraints:
            rounded = self._apply_rookie_constraints(rounded.astype(float), prepared)
            rounded = self._apply_general_constraints(rounded.astype(float), prepared, has_minimal_data)

        results = pd.DataFrame(
            {
                "predicted_position_raw": raw_predictions,
                "predicted_position": np.clip(np.round(rounded), 1, max(20, len(prepared))).astype(int),
                "original_index": prepared.index,
                "grid_position": prepared["grid_position"].to_numpy() if "grid_position" in prepared else DEFAULT_POSITION,
                "race_sort_score": self._race_sort_score(raw_predictions, rounded, prepared),
            }
        )
        results = results.sort_values(
            ["race_sort_score", "predicted_position", "predicted_position_raw", "grid_position", "original_index"]
        ).reset_index(drop=True)
        if apply_constraints:
            results = self._apply_post_sort_constraints(results, prepared)
        results["final_position"] = range(1, len(results) + 1)
        return PredictionFrame(results=results, prepared_features=prepared)

    def feature_importance(self) -> list[dict[str, Any]]:
        importances = getattr(self.model, "feature_importances_", np.zeros(len(self.feature_names)))
        rows = [
            {"feature": feature, "importance": round(float(importance), 6)}
            for feature, importance in zip(self.feature_names, importances)
        ]
        return sorted(rows, key=lambda item: item["importance"], reverse=True)

    def _prepare_features(self, x: pd.DataFrame, fit: bool) -> pd.DataFrame:
        frame = x.copy()
        for feature in self.feature_names:
            if feature not in frame:
                frame[feature] = np.nan
        frame = frame[self.feature_names].apply(pd.to_numeric, errors="coerce")

        if fit:
            self.fill_values = {}
            for feature in self.feature_names:
                values = frame[feature].replace([np.inf, -np.inf], np.nan).dropna()
                median = values.median() if not values.empty else None
                self.fill_values[feature] = _fallback_for_feature(feature, median)

        for feature in self.feature_names:
            frame[feature] = frame[feature].replace([np.inf, -np.inf], np.nan)
            frame[feature] = frame[feature].fillna(self.fill_values.get(feature, _fallback_for_feature(feature, None)))
        return frame

    def _has_minimal_data(self, x: pd.DataFrame) -> bool:
        performance_features = ["avg_lap_time", "best_lap_time", "avg_speed", "consistency_score"]
        meaningful = 0
        for feature in performance_features:
            if feature not in x:
                continue
            values = pd.to_numeric(x[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            meaningful += int(((values.notna()) & (values.abs() > 0.0001)).sum())
        return meaningful <= max(1, int(len(x) * 0.1))

    def _apply_grid_position_anchor(self, predictions: np.ndarray, x: pd.DataFrame) -> np.ndarray:
        if "grid_position" not in x:
            return predictions
        anchored = predictions.copy()
        grid_positions = pd.to_numeric(x["grid_position"], errors="coerce").fillna(DEFAULT_POSITION).to_numpy()
        strengths = self._team_strength_buckets(x)

        for index, prediction in enumerate(predictions):
            grid = grid_positions[index]
            strength = strengths[index]
            if grid <= 3:
                min_finish = max(1, grid - 2)
                max_finish = min(20, grid + (3 if strength == 0 else 5))
            elif grid <= 6:
                min_finish = max(1, grid - 3)
                max_finish = min(20, grid + (5 if strength <= 1 else 7))
            elif grid <= 10:
                min_finish = max(1, grid - 4)
                max_finish = min(20, grid + (7 if strength <= 1 else 9))
            else:
                min_finish = max(1, grid - (6 if strength <= 1 else 5))
                max_finish = min(20, grid + 10)
                if strength >= 1 and prediction < grid - 8:
                    prediction = grid - 6
            anchored[index] = np.clip(0.6 * grid + 0.4 * prediction, min_finish, max_finish)
        return anchored

    def _apply_general_constraints(self, predictions: np.ndarray, x: pd.DataFrame, has_minimal_data: bool) -> np.ndarray:
        if "grid_position" not in x:
            return predictions
        constrained = predictions.copy()
        grid_positions = pd.to_numeric(x["grid_position"], errors="coerce").fillna(DEFAULT_POSITION).to_numpy()
        strengths = self._team_strength_buckets(x)
        field_size = max(len(constrained), 20)

        for index, prediction in enumerate(constrained):
            grid = grid_positions[index]
            strength = strengths[index]
            if grid <= 3 and strength == 0 and prediction > 5:
                prediction = min(5, grid + 2)
            if grid <= 3 and prediction > 10:
                prediction = min(10, grid + 7)
            if grid <= 6 and strength == 0 and prediction > 8:
                prediction = min(8, grid + 2)
            if grid <= 10 and prediction > 15:
                prediction = min(15, grid + 5)
            if grid >= 14 and prediction < 4:
                prediction = max(10, prediction + 7)
            if grid >= 12 and strength >= 1 and prediction < 6:
                prediction = max(8, prediction + 3)

            improvement = grid - prediction
            if improvement > 10 or (has_minimal_data and improvement > 7):
                if grid >= 15:
                    max_improvement = 8
                elif grid >= 10:
                    max_improvement = 7
                else:
                    max_improvement = 6
                prediction = max(grid - max_improvement, prediction)

            loss = prediction - grid
            if loss > 10 and grid <= 5:
                prediction = min(grid + 8, prediction)
            constrained[index] = np.clip(prediction, 1, field_size)
        return constrained

    def _apply_rookie_constraints(self, predictions: np.ndarray, x: pd.DataFrame) -> np.ndarray:
        if "is_rookie" not in x or "grid_position" not in x:
            return predictions
        constrained = predictions.copy()
        rookies = pd.to_numeric(x["is_rookie"], errors="coerce").fillna(0).to_numpy() == 1
        if not np.any(rookies):
            return constrained
        grids = pd.to_numeric(x["grid_position"], errors="coerce").fillna(DEFAULT_POSITION).to_numpy()
        strengths = self._team_strength_buckets(x)
        for index, is_rookie in enumerate(rookies):
            if not is_rookie:
                continue
            grid = grids[index]
            prediction = constrained[index]
            strength = strengths[index]
            if strength == 0:
                max_finish = min(8, max(grid, 7))
            elif strength == 1:
                max_finish = min(12, max(grid, 10))
            else:
                max_finish = min(15, max(grid, 12))
            if prediction < max_finish:
                constrained[index] = min(max_finish, max(grid, 0.8 * grid + 0.2 * prediction))
            if grid > 10 and prediction < 6:
                constrained[index] = max(8, grid - 1)
            if grid > 5 and prediction < 4:
                constrained[index] = max(6, grid)
            if prediction < 2 and (grid > 1 or strength > 0):
                constrained[index] = max(4, grid)
        return np.clip(constrained, 1, max(20, len(constrained)))

    def _apply_post_sort_constraints(self, results: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
        if "is_rookie" not in x or "grid_position" not in results:
            return results
        rookie_lookup = pd.to_numeric(x["is_rookie"], errors="coerce").fillna(0).to_dict()
        target_positions: dict[int, float] = {}
        for sorted_index, row in results.iterrows():
            original_index = row["original_index"]
            is_rookie = rookie_lookup.get(original_index, 0) == 1
            grid = float(row.get("grid_position") or DEFAULT_POSITION)
            current_position = sorted_index + 1
            if is_rookie and grid >= 14 and current_position <= 10:
                target_positions[sorted_index] = max(14, grid)
            elif is_rookie and grid >= 12 and current_position <= 8:
                target_positions[sorted_index] = max(12, grid)
            elif is_rookie and grid >= 10 and current_position <= 5:
                target_positions[sorted_index] = max(10, grid)
            elif is_rookie and grid >= 8 and current_position <= 3:
                target_positions[sorted_index] = max(8, grid)
            elif grid >= 14 and current_position <= 3:
                target_positions[sorted_index] = max(10, grid)
        if not target_positions:
            return results
        ordered = sorted(
            range(len(results)),
            key=lambda index: (
                target_positions.get(index, index),
                float(results.iloc[index].get("race_sort_score") or 999.0),
                float(results.iloc[index]["predicted_position"]),
                float(results.iloc[index]["grid_position"]),
            ),
        )
        return pd.DataFrame([results.iloc[index] for index in ordered]).reset_index(drop=True)

    def _race_sort_score(self, raw_predictions: np.ndarray, constrained_predictions: np.ndarray, x: pd.DataFrame) -> np.ndarray:
        grid = pd.to_numeric(x.get("grid_position", DEFAULT_POSITION), errors="coerce").fillna(DEFAULT_POSITION).to_numpy()
        position_importance = pd.to_numeric(x.get("track_position_importance", 0.55), errors="coerce").fillna(0.55).to_numpy()
        overtaking = pd.to_numeric(x.get("track_overtaking_score", 0.5), errors="coerce").fillna(0.5).to_numpy()
        speed_score = pd.to_numeric(x.get("track_speed_score", 0.6), errors="coerce").fillna(0.6).to_numpy()
        driver_skill = pd.to_numeric(x.get("driver_skill_score", 0.08), errors="coerce").fillna(0.08).to_numpy()
        team_strength = pd.to_numeric(x.get("team_strength_score", 0.08), errors="coerce").fillna(0.08).to_numpy()
        track_exp = pd.to_numeric(x.get("driver_track_experience_score", 0.0), errors="coerce").fillna(0.0).to_numpy()
        estimated_grid_score = pd.to_numeric(x.get("estimated_grid_score", 0.0), errors="coerce").fillna(0.0).to_numpy()
        variation = pd.to_numeric(x.get("event_variation_score", 0.0), errors="coerce").fillna(0.0).to_numpy()
        rookie = pd.to_numeric(x.get("is_rookie", 0.0), errors="coerce").fillna(0.0).to_numpy()

        speed_fit = _rank_pct(pd.to_numeric(x.get("avg_speed", 200.0), errors="coerce").fillna(200.0).to_numpy())
        consistency_fit = _rank_pct(pd.to_numeric(x.get("consistency_score", 0.5), errors="coerce").fillna(0.5).to_numpy())
        base = np.asarray(raw_predictions, dtype=float) * (0.55 + position_importance * 0.15)
        grid_anchor = grid * (0.18 + position_importance * 0.28 - overtaking * 0.10)
        form_bonus = -(driver_skill * 9.0 + team_strength * 7.0 + estimated_grid_score * 4.0)
        circuit_bonus = -(track_exp * position_importance * 3.5 + speed_fit * speed_score * 1.8 + consistency_fit * (1.0 - overtaking) * 1.4)
        rookie_penalty = rookie * (0.8 + position_importance * 1.1)
        return base + grid_anchor + form_bonus + circuit_bonus + rookie_penalty - variation * 45.0

    def _team_strength_buckets(self, x: pd.DataFrame) -> np.ndarray:
        if "team_points_before_race" in x:
            points = pd.to_numeric(x["team_points_before_race"], errors="coerce").fillna(0).to_numpy()
            return np.where(points > 150, 0, np.where(points > 50, 1, 2))
        if "avg_finish_last_5_for_team" in x:
            avg_finish = pd.to_numeric(x["avg_finish_last_5_for_team"], errors="coerce").fillna(DEFAULT_POSITION).to_numpy()
            return np.where(avg_finish <= 5, 0, np.where(avg_finish <= 10, 1, 2))
        if "team_avg_finish_prior" in x:
            avg_finish = pd.to_numeric(x["team_avg_finish_prior"], errors="coerce").fillna(DEFAULT_POSITION).to_numpy()
            return np.where(avg_finish <= 6, 0, np.where(avg_finish <= 11, 1, 2))
        if "team_strength_score" in x:
            score = pd.to_numeric(x["team_strength_score"], errors="coerce").fillna(0.08).to_numpy()
            return np.where(score >= 0.13, 0, np.where(score >= 0.08, 1, 2))
        return np.ones(len(x), dtype=int) * 2


def _fallback_for_feature(feature: str, value: Any) -> float:
    try:
        if value is not None and not pd.isna(value):
            return float(value)
    except Exception:
        pass
    if "position" in feature or "finish" in feature or "grid" in feature:
        return DEFAULT_POSITION
    if "score" in feature or "rate" in feature or "percentile" in feature:
        return 0.5
    if "lap_time" in feature:
        return 95.0
    if "sector" in feature:
        return 30.0
    if "speed" in feature:
        return 200.0
    if "temp" in feature:
        return 25.0
    if feature == "reliability_score":
        return 1.0
    return 0.0


def _rank_pct(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    if series.nunique(dropna=True) <= 1:
        return np.ones(len(series)) * 0.5
    return series.rank(pct=True).to_numpy()
