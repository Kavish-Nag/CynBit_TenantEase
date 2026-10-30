from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_MODEL_PATH = ROOT / "data" / "custom_model.joblib"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, value))))


def make_synthetic_training_data(n_tenants: int = 420, cycles: int = 10, seed: int = 2026) -> pd.DataFrame:
    """Generate clearly labelled synthetic monthly payment histories for a reproducible demo."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for tenant_id in range(n_tenants):
        latent_risk = rng.betavariate(1.7, 3.2)
        rent_amount = rng.randrange(9000, 48001, 500)
        due_day = rng.choice([1, 3, 5, 7, 10, 15, 20, 25])
        lease_start_age = rng.randrange(2, 25)
        history: list[tuple[bool, int, bool]] = []  # late, delay, missed

        for cycle in range(cycles):
            observed = history
            last_n = observed[-6:]
            late_rate = sum(1 for x in last_n if x[0]) / len(last_n) if last_n else 0.0
            late_delays = [x[1] for x in last_n if x[0]]
            avg_delay = float(np.mean(late_delays)) if late_delays else 0.0
            streak = 0
            for item in reversed(observed):
                if item[0]:
                    streak += 1
                else:
                    break
            features = {
                "rent_amount": float(rent_amount),
                "previous_late_rate": float(late_rate),
                "avg_delay_days": float(avg_delay),
                "late_streak": float(streak),
                "payments_observed": float(len(observed)),
                "previous_missed_count": float(sum(1 for x in last_n if x[2])),
                "recent_delay_days": float(observed[-1][1]) if observed and observed[-1][0] else 0.0,
                "due_day": float(due_day),
                "lease_months": float(lease_start_age + cycle),
            }
            # Generate the next-cycle outcome from tenant-level and observable history signals.
            logit = (
                -2.75
                + latent_risk * 3.05
                + late_rate * 2.15
                + min(streak, 4) * 0.30
                + avg_delay * 0.055
                + features["previous_missed_count"] * 0.38
                + (due_day - 10) * 0.010
                + (0.16 if cycle in (4, 9) else 0.0)
            )
            probability = _sigmoid(logit)
            late = rng.random() < probability
            delay = rng.randint(1, 24) if late else 0
            missed = bool(late and rng.random() < 0.075)
            # Keep the training history outcome consistent with late/missed behavior.
            history.append((late, delay if late else 0, missed))
            if cycle >= 2:
                row = dict(features)
                row["late_next_cycle"] = int(late)
                row["tenant_group"] = tenant_id
                rows.append(row)
    return pd.DataFrame(rows)


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        metrics["roc_auc"] = 0.5
    return metrics


def _make_models() -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=320,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
            ]
        ),
    }


def train_and_evaluate(df: pd.DataFrame, group_col: str | None = "tenant_group") -> dict[str, Any]:
    missing = set(FEATURES + ["late_next_cycle"]) - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(sorted(missing))}")
    clean = df[FEATURES + ["late_next_cycle"] + ([group_col] if group_col and group_col in df.columns else [])].copy()
    for col in FEATURES + ["late_next_cycle"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    clean["late_next_cycle"] = clean["late_next_cycle"].astype(int)
    clean = clean[clean["late_next_cycle"].isin([0, 1])]
    if len(clean) < 80:
        raise ValueError("At least 80 valid labelled rows are required to train a custom model.")
    if clean["late_next_cycle"].nunique() < 2:
        raise ValueError("Training data must contain both on-time (0) and late (1) outcomes.")

    X = clean[FEATURES].astype(float)
    y = clean["late_next_cycle"].to_numpy()
    if group_col and group_col in clean.columns and clean[group_col].nunique() >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=clean[group_col]))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(clean)), test_size=0.25, random_state=42, stratify=y
        )
    results: dict[str, Any] = {}
    fitted = {}
    for name, model in _make_models().items():
        model.fit(X.iloc[train_idx], y[train_idx])
        probabilities = model.predict_proba(X.iloc[test_idx])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results[name] = _metrics(y[test_idx], probabilities, predictions)
        fitted[name] = model
    chosen = max(results, key=lambda name: (results[name]["pr_auc"], -results[name]["brier"]))
    final_model = _make_models()[chosen]
    final_model.fit(X, y)
    return {
        "model": final_model,
        "model_name": chosen,
        "metrics": results,
        "rows": int(len(clean)),
        "positive_rate": float(y.mean()),
        "source": "uploaded labelled dataset",
        "feature_importances": _feature_importance(final_model, chosen),
    }


def _feature_importance(model: Any, model_name: str) -> dict[str, float]:
    if model_name == "Random Forest":
        values = model.feature_importances_
    elif model_name == "Logistic Regression":
        values = np.abs(model.named_steps["model"].coef_[0])
    else:
        return {}
    total = float(np.sum(values)) or 1.0
    return {name: float(value / total) for name, value in zip(FEATURES, values)}


def get_default_bundle() -> dict[str, Any]:
    """Train a reproducible demo model, evaluating by tenant group to reduce leakage."""
    df = make_synthetic_training_data()
    bundle = train_and_evaluate(df, group_col="tenant_group")
    bundle["source"] = "synthetic demonstration data"
    bundle["synthetic_rows"] = len(df)
    return bundle


def get_active_bundle() -> dict[str, Any]:
    if CUSTOM_MODEL_PATH.exists():
        try:
            bundle = joblib.load(CUSTOM_MODEL_PATH)
            if isinstance(bundle, dict) and "model" in bundle:
                return bundle
        except Exception:
            # Fall back to a reproducible default instead of blocking the app.
            pass
    return get_default_bundle()


def save_custom_bundle(bundle: dict[str, Any]) -> None:
    CUSTOM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, CUSTOM_MODEL_PATH)


def create_training_template() -> bytes:
    df = make_synthetic_training_data(n_tenants=90, cycles=9, seed=2026)
    # A small labelled example file is included to show the expected schema.
    return df.to_csv(index=False).encode("utf-8")


def score_features(bundle: dict[str, Any], feature_rows: list[dict[str, float]]) -> np.ndarray:
    if not feature_rows:
        return np.array([])
    X = pd.DataFrame(feature_rows, columns=FEATURES).astype(float)
    return bundle["model"].predict_proba(X)[:, 1]


def sample_training_dataframe() -> pd.DataFrame:
    return make_synthetic_training_data()


def feature_importance_for_bundle(bundle: dict[str, Any]) -> dict[str, float]:
    return bundle.get("feature_importances") or _feature_importance(bundle["model"], bundle["model_name"])
