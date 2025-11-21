import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# Seed NumPy for reproducible ML experiments
np.random.seed(42)


class PollutionModel:
    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        self.model = self._create_model(model_type)

    def _create_model(self, model_type: str):
        # Factory for supported regression models
        mt = model_type.lower()
        if mt in ("rf", "random_forest"):
            return RandomForestRegressor(n_estimators=200, random_state=42)
        if mt in ("gb", "gbr", "gradient_boosting"):
            return GradientBoostingRegressor(random_state=42)
        raise ValueError(f"Unsupported model_type: {model_type}")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        # Prepare features and targets from time-series style data
        if "step" in df.columns:
            df = df.sort_values("step").reset_index(drop=True)
        feature_cols: List[str] = [
            c
            for c in [
                "total_pollution",
                "mean_pollution",
                "max_pollution",
                "alerts",
                "mitigated_amount",
            ]
            if c in df.columns
        ]
        if not feature_cols:
            raise ValueError("No valid feature columns found in data")
        X = df[feature_cols].values
        y = None
        if "target_pollution" in df.columns:
            y = df["target_pollution"].values
        elif "total_pollution" in df.columns and len(df) > 1:
            shifted = df["total_pollution"].shift(-1)
            X = X[:-1, :]
            y = shifted[:-1].values
        else:
            raise ValueError("Cannot infer target variable from data")
        return X, y, feature_cols

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        # Train the model and report evaluation metrics
        X, y, _ = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        # Predict pollution outcome for given feature vectors
        return self.model.predict(features)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        # Map model feature importances back to names
        importances: Dict[str, float] = {}
        if hasattr(self.model, "feature_importances_"):
            values = list(self.model.feature_importances_)
            for name, val in zip(feature_names, values):
                importances[name] = float(val)
        return importances

    def save_model(self, path: str) -> None:
        # Persist trained model to disk
        payload = {"model_type": self.model_type, "model": self.model}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load_model(path: str) -> "PollutionModel":
        # Load a previously saved model
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model = PollutionModel(model_type=payload["model_type"])
        model.model = payload["model"]
        return model


def history_json_to_dataframe(path: str) -> pd.DataFrame:
    # Convert simulation JSON history to a tabular DataFrame
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("history", [])
    if not history:
        raise ValueError(f"No history entries found in {path}")
    return pd.DataFrame(history)


def convert_kaggle_csv_to_json(csv_path: str, json_path: str) -> None:
    # Convert a Kaggle CSV into the project JSON history format
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("Kaggle CSV must contain at least one numeric column")
    cols = list(numeric.columns)
    total = numeric[cols[0]]
    mean = numeric[cols[1]] if len(cols) > 1 else total
    max_vals = numeric[cols[2]] if len(cols) > 2 else total
    steps = list(range(len(total)))
    alerts = [0.0] * len(total)
    mitigated = [0.0] * len(total)
    history: List[Dict[str, Any]] = []
    for i, t in enumerate(total):
        entry = {
            "step": int(steps[i]),
            "total_pollution": float(t),
            "mean_pollution": float(mean.iloc[i]),
            "max_pollution": float(max_vals.iloc[i]),
            "alerts": float(alerts[i]),
            "mitigated_amount": float(mitigated[i]),
        }
        history.append(entry)
    payload = {"source": "kaggle_csv", "csv_file": os.path.basename(csv_path), "history": history}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_from_history(json_path: str, model_type: str = "rf") -> Tuple[PollutionModel, Dict[str, float], List[str]]:
    # Train a model directly from simulation or converted JSON
    df = history_json_to_dataframe(json_path)
    model = PollutionModel(model_type=model_type)
    metrics = model.train(df)
    X, _, feature_names = model.prepare_features(df)
    importance = model.get_feature_importance(feature_names)
    for name, val in importance.items():
        print(f"Feature importance - {name}: {val:.4f}")
    return model, metrics, feature_names


def train_from_kaggle(csv_file: str, model_type: str = "rf") -> Tuple[PollutionModel, Dict[str, float], List[str], str]:
    # Train a model from a Kaggle CSV by converting to JSON first
    base, _ = os.path.splitext(csv_file)
    json_path = base + "_converted.json"
    convert_kaggle_csv_to_json(csv_file, json_path)
    model, metrics, feature_names = train_from_history(json_path, model_type=model_type)
    return model, metrics, feature_names, json_path


def main() -> None:
    # CLI: python train_model.py <csv_file> [model_type]
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <csv_file> [model_type]")
        print("Example: python train_model.py data/air_quality.csv rf")
        return
    csv_file = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "rf"
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    model, metrics, feature_names, json_path = train_from_kaggle(csv_file, model_type=model_type)
    base, _ = os.path.splitext(csv_file)
    model_path = f"{base}_{model_type}_model.pkl"
    model.save_model(model_path)
    print(f"Training data JSON: {json_path}")
    print(f"Saved trained {model_type} model to {model_path}")
    print("Metrics:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R2 : {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
