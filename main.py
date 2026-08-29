"""
Heat-health forecasting system with separate training and inference.

Primary prediction target
    6-hour-ahead maximum Human Thermal Stress Index, or HTSI.

Why this target
    It is more suitable for live API inference than a 3-day target because
    the API can score the next six hours immediately after new observations
    or forecast rows arrive. A separate deterministic 3-day outlook is also
    returned when three future dates are present in the input.

The supplied vertically merged weather file has ten rows per timestamp and
no location column. The training loader treats row position within each
timestamp as an anonymous but stable series only after validating that:
    - every timestamp has exactly ten rows
    - row order is stable
    - each anonymous series has one row per timestamp

For production, pass a stable location_id from the data provider. Do not
reinterpret anonymous series IDs as real city names.

Install
    pip install pandas numpy scikit-learn xgboost joblib tqdm pythermalcomfort
    pip install fastapi uvicorn pydantic

Training
    python heat_health_system_v2.py train

API
    uvicorn heat_health_system_v2:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health
    GET  /model-info
    POST /predict

POST /predict accepts either:
    {"records": [{...}, {...}]}
or a bare list of weather records.

Required live fields
    datetime
    location_id
    temperature_2m
    relative_humidity_2m
    wind_speed_10m
    shortwave_radiation

No city, latitude, longitude, or vulnerability field is used as a model
feature. Geographic fields may be carried as metadata but are excluded.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Versioned configuration
# ---------------------------------------------------------------------

MODEL_VERSION = "heat-health-htsi-6h-v2.0.0"
PREPROCESSING_VERSION = "thermal-features-v2.0.0"

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "weather_all_cities_vertical.csv"
MODEL_DIR = BASE_DIR / "heat_health_model"
MODEL_FILE = MODEL_DIR / f"{MODEL_VERSION}.joblib"
SCHEMA_FILE = MODEL_DIR / f"{MODEL_VERSION}.schema.json"
HISTORY_FILE = MODEL_DIR / "inference_history.csv"
PREDICTIONS_FILE = MODEL_DIR / "latest_predictions.csv"

HTSI_THRESHOLD = 70.0
TARGET_HORIZON_HOURS = 6
ROLLING_WINDOWS_HOURS = (6, 12, 24, 48, 72)
ANONYMOUS_SERIES_COUNT = 10
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------

MODEL_FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "radiation",
    "dew_point",
    "apparent_temperature",
    "wet_bulb_temperature",
    "heat_index",
    "wbgt",
    "utci",
    "heat_index_norm",
    "wbgt_norm",
    "utci_norm",
    "htsi",
    "htsi_6h_mean",
    "htsi_12h_mean",
    "htsi_24h_mean",
    "htsi_48h_mean",
    "htsi_72h_mean",
    "temperature_6h_mean",
    "humidity_6h_mean",
    "wind_speed_6h_mean",
    "radiation_6h_mean",
    "consecutive_hot_hours",
    "temperature_anomaly",
    "htsi_anomaly",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

WEATHER_COLUMNS = [
    "datetime",
    "location_id",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "dew_point_2m",
    "apparent_temperature",
    "wet_bulb_temperature_2m",
]


# ---------------------------------------------------------------------
# Calculation helpers
# ---------------------------------------------------------------------

def to_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def wet_bulb_stull(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    rh = rh_pct.clip(1, 100)
    return (
        temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def heat_index_celsius(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    temp_f = temp_c * 9 / 5 + 32
    rh = rh_pct.clip(0, 100)
    simple_f = 0.5 * (temp_f + 61 + (temp_f - 68) * 1.2 + rh * 0.094)
    rothfusz_f = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh
        - 0.22475541 * temp_f * rh
        - 0.00683783 * temp_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * temp_f**2 * rh
        + 0.00085282 * temp_f * rh**2
        - 0.00000199 * temp_f**2 * rh**2
    )
    return pd.Series(
        np.where(rothfusz_f < 80, simple_f, rothfusz_f) * 5 / 9 - 160 / 9,
        index=temp_c.index,
    )


def outdoor_wbgt_proxy(
    temp_c: pd.Series,
    wet_bulb_c: pd.Series,
    radiation: pd.Series,
    wind: pd.Series,
) -> pd.Series:
    wind_safe = wind.clip(lower=0.5).fillna(0.5)
    radiation_safe = radiation.clip(0, 1200).fillna(0)
    globe_c = temp_c + 0.12 * np.sqrt(radiation_safe) / np.sqrt(wind_safe)
    return 0.7 * wet_bulb_c + 0.2 * globe_c + 0.1 * temp_c


def calculate_utci(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    temp = frame["temperature"]
    rh = frame["humidity"].clip(0, 100)
    wind = frame["wind_speed"].clip(lower=0.5)
    mean_radiant = temp + 0.15 * np.sqrt(frame["radiation"].clip(0, 1200).fillna(0))

    try:
        from pythermalcomfort.models import utci as utci_model

        values: list[float] = []
        for start in tqdm(range(0, len(frame), 50_000), desc="Calculating UTCI"):
            stop = min(start + 50_000, len(frame))
            result = utci_model(
                tdb=temp.iloc[start:stop].to_numpy(),
                tr=mean_radiant.iloc[start:stop].to_numpy(),
                v=wind.iloc[start:stop].to_numpy(),
                rh=rh.iloc[start:stop].to_numpy(),
                units="SI",
                limit_inputs=False,
                round_output=False,
            )
            values.extend(np.asarray(result.utci, dtype=float))
        return pd.Series(values, index=frame.index), "pythermalcomfort"
    except Exception as error:
        print(f"UTCI fallback used - {error}")
        approximate = temp + 0.20 * (rh - 50) - 0.70 * (wind - 1) + 0.015 * frame["radiation"]
        return approximate, "fallback_approximation"


def minmax_stress(values: pd.Series, low: float, high: float) -> pd.Series:
    return ((values - low) / (high - low) * 100).clip(0, 100)


def run_length(flags: pd.Series) -> pd.Series:
    result = np.zeros(len(flags), dtype=float)
    current = 0
    for index, flag in enumerate(flags.to_numpy()):
        current = current + 1 if flag == 1 else 0
        result[index] = current
    return pd.Series(result, index=flags.index)


def safe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


# ---------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------

def assign_anonymous_series(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Correctly handles the vertically merged ten-series file.

    The source has no city identifiers. We preserve row position within each
    timestamp as series_00 ... series_09, but validate the structure first.
    """
    out = frame.copy()
    counts = out.groupby("datetime").size()
    if counts.empty:
        raise ValueError("No valid timestamps found.")
    if counts.nunique() != 1 or int(counts.iloc[0]) != ANONYMOUS_SERIES_COUNT:
        raise ValueError(
            "Anonymous vertical-series inference requires exactly "
            f"{ANONYMOUS_SERIES_COUNT} rows at every timestamp. "
            f"Observed counts - {counts.value_counts().to_dict()}"
        )
    out = out.sort_values(["datetime"]).reset_index(drop=True)
    out["location_id"] = (
        "series_"
        + out.groupby("datetime").cumcount().astype(str).str.zfill(2)
    )
    return out


def normalize_weather_input(
    records: pd.DataFrame,
    allow_anonymous: bool = False,
) -> pd.DataFrame:
    out = records.copy()
    if "datetime" not in out.columns:
        raise ValueError("Input requires datetime.")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=False)
    out = out.dropna(subset=["datetime"]).copy()

    if "location_id" not in out.columns:
        if not allow_anonymous:
            raise ValueError("Live inference requires a stable location_id.")
        out = assign_anonymous_series(out)
    out["location_id"] = out["location_id"].astype(str)

    required = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"Missing weather fields - {missing}")

    out = out.sort_values(["location_id", "datetime"]).drop_duplicates(
        ["location_id", "datetime"], keep="last"
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# Thermal feature engineering
# ---------------------------------------------------------------------

def calculate_thermal_features(
    records: pd.DataFrame,
    historical_context: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    current = normalize_weather_input(records, allow_anonymous=False)
    frames = []
    if historical_context is not None and len(historical_context):
        frames.append(normalize_weather_input(historical_context, allow_anonymous=False))
    frames.append(current)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["location_id", "datetime"])
    combined = combined.drop_duplicates(["location_id", "datetime"], keep="last")
    combined = combined.reset_index(drop=True)

    combined["temperature"] = to_numeric(combined, "temperature_2m")
    combined["humidity"] = to_numeric(combined, "relative_humidity_2m").clip(0, 100)
    combined["wind_speed"] = to_numeric(combined, "wind_speed_10m").clip(lower=0)
    combined["radiation"] = to_numeric(combined, "shortwave_radiation").clip(lower=0).fillna(0)
    combined["dew_point"] = to_numeric(combined, "dew_point_2m")
    combined["apparent_temperature"] = to_numeric(combined, "apparent_temperature")
    combined["wet_bulb_temperature"] = to_numeric(combined, "wet_bulb_temperature_2m")

    combined["wet_bulb_temperature"] = combined["wet_bulb_temperature"].fillna(
        wet_bulb_stull(combined["temperature"], combined["humidity"])
    )
    combined["heat_index"] = heat_index_celsius(combined["temperature"], combined["humidity"])
    combined["wbgt"] = outdoor_wbgt_proxy(
        combined["temperature"],
        combined["wet_bulb_temperature"],
        combined["radiation"],
        combined["wind_speed"],
    )
    combined["utci"], combined["utci_method"] = calculate_utci(combined)

    combined["heat_index_norm"] = minmax_stress(combined["heat_index"], 26.7, 54.0)
    combined["wbgt_norm"] = minmax_stress(combined["wbgt"], 18.0, 35.0)
    combined["utci_norm"] = minmax_stress(combined["utci"], 9.0, 46.0)
    combined["htsi"] = (
        0.30 * combined["heat_index_norm"]
        + 0.35 * combined["wbgt_norm"]
        + 0.35 * combined["utci_norm"]
    ).clip(0, 100)

    combined["date"] = combined["datetime"].dt.floor("D")
    combined["hour"] = combined["datetime"].dt.hour
    combined["day_of_year"] = combined["datetime"].dt.dayofyear
    combined["hot_hour_flag"] = (combined["htsi"] >= HTSI_THRESHOLD).astype(int)

    grouped = combined.groupby("location_id", group_keys=False)
    combined["consecutive_hot_hours"] = grouped["hot_hour_flag"].apply(run_length).reset_index(
        level=0, drop=True
    )

    for hours in ROLLING_WINDOWS_HOURS:
        combined[f"htsi_{hours}h_mean"] = grouped["htsi"].transform(
            lambda values: values.rolling(hours, min_periods=1).mean()
        )
    for column in ["temperature", "humidity", "wind_speed", "radiation"]:
        combined[f"{column}_6h_mean"] = grouped[column].transform(
            lambda values: values.rolling(6, min_periods=1).mean()
        )

    combined["temperature_climatology"] = combined.groupby(
        ["location_id", "hour", combined["datetime"].dt.month]
    )["temperature"].transform("median")
    combined["htsi_climatology"] = combined.groupby(
        ["location_id", "hour", combined["datetime"].dt.month]
    )["htsi"].transform("median")
    combined["temperature_anomaly"] = (
        combined["temperature"] - combined["temperature_climatology"]
    ).fillna(0)
    combined["htsi_anomaly"] = (
        combined["htsi"] - combined["htsi_climatology"]
    ).fillna(0)

    combined["hour_sin"] = np.sin(2 * np.pi * combined["hour"] / 24)
    combined["hour_cos"] = np.cos(2 * np.pi * combined["hour"] / 24)
    combined["day_of_year_sin"] = np.sin(2 * np.pi * combined["day_of_year"] / 365.25)
    combined["day_of_year_cos"] = np.cos(2 * np.pi * combined["day_of_year"] / 365.25)

    return combined[combined["datetime"].isin(current["datetime"])].copy()


def add_6h_target(features: pd.DataFrame) -> pd.DataFrame:
    out = features.sort_values(["location_id", "datetime"]).copy()
    out["target_datetime"] = out["datetime"] + pd.Timedelta(hours=TARGET_HORIZON_HOURS)
    future = out[["location_id", "datetime", "htsi"]].rename(
        columns={"datetime": "target_datetime", "htsi": "target_htsi_6h"}
    )
    out = out.merge(future, on=["location_id", "target_datetime"], how="left")
    out["target_hot_6h"] = np.where(
        out["target_htsi_6h"].notna(),
        (out["target_htsi_6h"] >= HTSI_THRESHOLD).astype(int),
        np.nan,
    )
    return out


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_FILE, encoding="ascii")
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
    raw = raw.dropna(subset=["datetime"]).copy()
    if "location_id" not in raw.columns:
        raw = assign_anonymous_series(raw)
    else:
        raw["location_id"] = raw["location_id"].astype(str)
    return raw


def train_model() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_training_data()
    features = calculate_thermal_features(raw)
    training = add_6h_target(features).dropna(subset=MODEL_FEATURES + ["target_hot_6h"])
    training = training.sort_values("datetime").reset_index(drop=True)

    unique_dates = np.sort(training["date"].unique())
    if len(unique_dates) < 10:
        raise ValueError("At least ten unique dates are required for training.")
    cutoff = unique_dates[int(len(unique_dates) * 0.80)]
    train = training[training["date"] < cutoff]
    test = training[training["date"] >= cutoff]
    if train["target_hot_6h"].nunique() < 2:
        raise ValueError(
            "The training target has one class only. Adjust HTSI_THRESHOLD or add more data."
        )

    from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

    try:
        from xgboost import XGBClassifier
        model_class = "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        XGBClassifier = HistGradientBoostingClassifier
        model_class = "sklearn_hist_gradient_boosting"

    positive_count = train["target_hot_6h"].sum()
    negative_count = len(train) - positive_count
    if model_class == "xgboost":
        model = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negative_count / max(positive_count, 1),
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
    else:
        model = XGBClassifier(
            max_iter=350,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    model.fit(train[MODEL_FEATURES], train["target_hot_6h"].astype(int))

    test_probability = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    test_prediction = (test_probability >= 0.50).astype(int)
    metrics: dict[str, Any] = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(train["target_hot_6h"].mean()),
        "test_positive_rate": float(test["target_hot_6h"].mean()),
        "cutoff_date": str(pd.Timestamp(cutoff).date()),
    }
    if test["target_hot_6h"].nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(test["target_hot_6h"], test_probability))
        metrics["average_precision"] = float(
            average_precision_score(test["target_hot_6h"], test_probability)
        )
        print(classification_report(
            test["target_hot_6h"].astype(int),
            test_prediction,
            zero_division=0,
        ))

    schema = {
        "model_version": MODEL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "target": "target_hot_6h",
        "target_definition": "HTSI at exactly datetime plus six hours is at least HTSI_THRESHOLD",
        "target_horizon_hours": TARGET_HORIZON_HOURS,
        "htsi_threshold": HTSI_THRESHOLD,
        "features": MODEL_FEATURES,
        "excluded_from_model": ["city", "ward_name", "latitude", "longitude", "vulnerability_score"],
        "metrics": metrics,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "utci_method": str(features["utci_method"].iloc[0]),
        "model_algorithm": model_class,
    }
    SCHEMA_FILE.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": model,
            "schema": schema,
            "feature_columns": MODEL_FEATURES,
            "model_version": MODEL_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
        },
        MODEL_FILE,
    )

    print(f"Model saved - {MODEL_FILE}")
    print(f"Schema saved - {SCHEMA_FILE}")
    print(json.dumps(metrics, indent=2))
    return {"model": model, "schema": schema}


# ---------------------------------------------------------------------
# Inference state and live predictions
# ---------------------------------------------------------------------

def load_saved_bundle() -> dict[str, Any]:
    if not MODEL_FILE.exists() or not SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"Saved model not found. Run `python {Path(__file__).name} train` first."
        )
    bundle = joblib.load(MODEL_FILE)
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    if bundle["feature_columns"] != schema["features"]:
        raise ValueError("Model feature schema and saved schema do not match.")
    return bundle


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(HISTORY_FILE, parse_dates=["datetime"])


def update_history(records: pd.DataFrame, max_hours: int = 120) -> pd.DataFrame:
    historical = load_history()
    combined = pd.concat([historical, records], ignore_index=True)
    combined = combined.drop_duplicates(["location_id", "datetime"], keep="last")
    combined = combined.sort_values(["location_id", "datetime"])
    combined = combined.groupby("location_id", group_keys=False).tail(max_hours)
    combined.to_csv(HISTORY_FILE, index=False)
    return combined.reset_index(drop=True)


def build_prediction_response(records: pd.DataFrame) -> dict[str, Any]:
    bundle = load_saved_bundle()
    current = normalize_weather_input(records, allow_anonymous=False)
    historical = load_history()
    context = calculate_thermal_features(current, historical)
    latest = context.sort_values("datetime").groupby("location_id").tail(1).copy()
    probability = bundle["model"].predict_proba(latest[MODEL_FEATURES])[:, 1]

    latest["probability_hot_6h"] = probability
    latest["predicted_hot_6h"] = (probability >= 0.50).astype(int)
    latest["risk_index"] = (
        100 * (
            0.80 * latest["probability_hot_6h"]
            + 0.20 * latest["htsi"] / 100
        )
    ).clip(0, 100)
    latest["risk_category"] = pd.cut(
        latest["risk_index"],
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype(str)
    latest["recommended_action"] = latest["risk_category"].map({
        "Low": "Routine monitoring",
        "Moderate": "Hydration, shade, and vulnerable-person advisory",
        "High": "Open cooling centers and shift outdoor work",
        "Very High": "Activate heat action plan and emergency staffing",
    })

    # Optional deterministic three-day outlook if future forecast rows exist.
    daily = context.groupby(["location_id", "date"], as_index=False).agg(
        daily_max_htsi=("htsi", "max")
    )
    three_day = []
    for location_id, group in daily.groupby("location_id"):
        group = group.sort_values("date")
        future = group.tail(3)
        three_day.append({
            "location_id": location_id,
            "next_3_day_max_htsi": safe_float(future["daily_max_htsi"].max()),
            "next_3_day_heatwave_flag": bool(
                len(future) >= 3 and future["daily_max_htsi"].max() >= HTSI_THRESHOLD
            ),
        })
    three_day_df = pd.DataFrame(three_day)
    latest = latest.merge(three_day_df, on="location_id", how="left")

    update_history(current)
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(PREDICTIONS_FILE, index=False)

    output_columns = [
        "location_id", "datetime", "temperature", "humidity", "wind_speed",
        "radiation", "heat_index", "wbgt", "utci", "htsi",
        "probability_hot_6h", "predicted_hot_6h", "risk_index",
        "risk_category", "recommended_action", "next_3_day_max_htsi",
        "next_3_day_heatwave_flag",
    ]
    latest = latest[output_columns]
    latest = latest.replace({np.nan: None})
    return {
        "model_version": bundle["model_version"],
        "target": "6-hour-ahead HTSI threshold exceedance",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "predictions": latest.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------
# Pydantic and FastAPI layer
# ---------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, ConfigDict

    class WeatherRecord(BaseModel):
        model_config = ConfigDict(extra="allow")
        datetime: datetime
        location_id: str = Field(min_length=1)
        temperature_2m: float
        relative_humidity_2m: float = Field(ge=0, le=100)
        wind_speed_10m: float = Field(ge=0)
        shortwave_radiation: float = Field(ge=0)
        dew_point_2m: Optional[float] = None
        apparent_temperature: Optional[float] = None
        wet_bulb_temperature_2m: Optional[float] = None

    class PredictionRequest(BaseModel):
        records: list[WeatherRecord] = Field(min_length=1, max_length=10_000)

    app = FastAPI(
        title="India Heat-health Early Warning API",
        version=MODEL_VERSION,
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_available": MODEL_FILE.exists(),
            "model_version": MODEL_VERSION,
        }

    @app.get("/model-info")
    def model_info():
        try:
            bundle = load_saved_bundle()
            return {
                "model_version": bundle["model_version"],
                "preprocessing_version": bundle["preprocessing_version"],
                "schema": bundle["schema"],
            }
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error))

    @app.post("/predict")
    def predict(request: PredictionRequest):
        try:
            records = pd.DataFrame([item.model_dump() for item in request.records])
            return build_prediction_response(records)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error))

except Exception:
    app = None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        train_model()
    else:
        print("Use one of these commands")
        print(f"python {Path(__file__).name} train")
        print(f"uvicorn {Path(__file__).stem}:app --host 0.0.0.0 --port 8000")
