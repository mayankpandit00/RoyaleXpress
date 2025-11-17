# ml_utils.py
"""
ML utilities for RoyaleXpress.

- load_data() takes no args and loads users, products, orders, riders, feedback
  from the package-local 'data' folder.
- load_models() loads joblib models from the package-local 'models' folder.
- All heavy imports (pandas, joblib, numpy, sklearn, etc.) are done inside
  functions to avoid ImportError at module import time.
- check_requirements() returns a list of missing packages (useful for UI).
"""

import os
from typing import Tuple, Dict, Any, List

# Directories (package-local)
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHART_DIR = os.path.join(BASE_DIR, "charts")


def check_requirements() -> List[str]:
    """
    Return a list of missing package names required for ML ops.
    This is a lightweight check (non-exhaustive).
    """
    required = ("joblib", "sklearn", "pandas", "numpy")
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    return missing


# ---------------------------
# Data loading (no-arg)
# ---------------------------
def load_data() -> Tuple[Any, Any, Any, Any, Any]:
    """
    Load users, products, orders, riders, feedback from DATA_DIR.

    Returns:
        tuple: (users, products, orders, riders, feedback) as pandas.DataFrame objects.

    Raises:
        ImportError: if pandas is not installed.
        FileNotFoundError: if expected CSV files are missing (with helpful message).
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required to load data. Add 'pandas' to requirements.txt.") from e

    # helper to locate a CSV file in DATA_DIR or repo root fallback
    def _find_csv(name: str) -> str:
        cand1 = os.path.join(DATA_DIR, name)
        cand2 = os.path.join(BASE_DIR, name)
        cand3 = os.path.join(".", name)
        for c in (cand1, cand2, cand3):
            if os.path.exists(c):
                return c
        # not found
        return ""

    files = {
        "users": _find_csv("users.csv"),
        "products": _find_csv("products.csv"),
        "orders": _find_csv("orders.csv"),
        "riders": _find_csv("riders.csv"),
        "feedback": _find_csv("feedback.csv"),
    }

    missing = [k for k, v in files.items() if not v]
    if missing:
        # Provide a clear message but still try to return empty DataFrames for usability
        # Caller (e.g. Streamlit) can choose to warn the user.
        msg = f"Missing CSV files for: {', '.join(missing)}. Checked {DATA_DIR}, {BASE_DIR}, and project root."
        # Build empty dataframes for missing ones and continue
        dfs = {}
        for k, path in files.items():
            if path:
                try:
                    dfs[k] = pd.read_csv(path)
                except Exception as e:
                    raise RuntimeError(f"Failed reading {path}: {e}") from e
            else:
                dfs[k] = pd.DataFrame()
        # attach message as attribute for optional UI display
        dfs["_load_warning"] = msg
        return dfs["users"], dfs["products"], dfs["orders"], dfs["riders"], dfs["feedback"]

    # All files found — load them
    try:
        users = pd.read_csv(files["users"])
        products = pd.read_csv(files["products"])
        orders = pd.read_csv(files["orders"])
        riders = pd.read_csv(files["riders"])
        feedback = pd.read_csv(files["feedback"])
    except Exception as e:
        raise RuntimeError(f"Error reading CSV files: {e}") from e

    return users, products, orders, riders, feedback


# ---------------------------
# Model loading
# ---------------------------
def load_models() -> Dict[str, Any]:
    """
    Load joblib models from MODEL_DIR and return a dict of models.

    Returns dict keys:
      'eta', 'highvalue', 'kmeans', 'kmeans_scaler', 'rider', 'sentiment'

    Behavior:
      - If joblib is missing, raises ImportError with actionable message.
      - If model file missing, sets that key to None and continues.
      - If loading fails for a file, raises RuntimeError.

    """
    try:
        import joblib
    except ImportError as e:
        raise ImportError("joblib is required to load models. Add 'joblib' to requirements.txt.") from e

    model_map = {
        "eta": os.path.join(MODEL_DIR, "eta_gbr_pipeline.joblib"),
        "highvalue": os.path.join(MODEL_DIR, "highvalue_rf_pipeline.joblib"),
        "kmeans": os.path.join(MODEL_DIR, "kmeans_users.joblib"),
        "kmeans_scaler": os.path.join(MODEL_DIR, "kmeans_scaler.joblib"),
        "rider": os.path.join(MODEL_DIR, "riderperf_rf_pipeline.joblib"),
        "sentiment": os.path.join(MODEL_DIR, "sentiment_tfidf_lr.joblib"),
    }

    models: Dict[str, Any] = {}
    for key, path in model_map.items():
        if not os.path.exists(path):
            # not fatal — return None for this key so callers can decide
            models[key] = None
            continue
        try:
            models[key] = joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{key}' from {path}: {e}") from e

    return models


# ---------------------------
# Prediction helpers (use models dict)
# ---------------------------
def predict_eta(models: Dict[str, Any], distance_km, traffic_level, avg_speed_kmph, loyalty_score, order_value):
    """
    Predict ETA given the inputs. Expects models['eta'] to be a pipeline that accepts a DataFrame.
    Returns float minutes.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for predict_eta().") from e

    eta_model = models.get("eta") if isinstance(models, dict) else None
    if eta_model is None:
        raise RuntimeError("ETA model not loaded (models['eta'] is None).")

    df = pd.DataFrame([{
        "distance_km": float(distance_km),
        "traffic_level": str(traffic_level),
        "avg_speed_kmph": float(avg_speed_kmph),
        "loyalty_score": float(loyalty_score),
        "order_value": float(order_value),
    }])
    pred = eta_model.predict(df)
    return float(pred[0])


def predict_highvalue(models: Dict[str, Any], age, loyalty_score, order_count, avg_order_value, avg_delivery_time):
    """
    Predict if user is high-value. Returns (int_pred, proba_or_None).
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for predict_highvalue().") from e

    hv_model = models.get("highvalue") if isinstance(models, dict) else None
    if hv_model is None:
        raise RuntimeError("High-value model not loaded (models['highvalue'] is None).")

    df = pd.DataFrame([{
        "age": float(age),
        "loyalty_score": float(loyalty_score),
        "order_count": float(order_count),
        "avg_order_value": float(avg_order_value),
        "avg_delivery_time": float(avg_delivery_time),
    }])
    pred = hv_model.predict(df)
    proba = None
    if hasattr(hv_model, "predict_proba"):
        proba = float(hv_model.predict_proba(df)[0, 1])
    return int(pred[0]), proba


def assign_user_cluster(models: Dict[str, Any], user_row: dict) -> int:
    """
    Assign cluster for a single user row (dict or pandas Series-like).
    Uses kmeans_scaler and kmeans model.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for assign_user_cluster().") from e

    scaler = models.get("kmeans_scaler")
    kmeans = models.get("kmeans")
    if scaler is None or kmeans is None:
        raise RuntimeError("KMeans scaler/model not loaded (models['kmeans_scaler'] or models['kmeans'] is None).")

    # Extract features safely
    def _val(d, key, fallback=0.0):
        try:
            return float(d.get(key, fallback))
        except Exception:
            return float(fallback)

    arr = np.array([[
        _val(user_row, "age", 0.0),
        _val(user_row, "avg_monthly_spend", 0.0),
        _val(user_row, "loyalty_score", 0.0),
        _val(user_row, "order_count", 0.0),
    ]])
    scaled = scaler.transform(arr)
    cluster = int(kmeans.predict(scaled)[0])
    return cluster


def predict_rider_rating(models: Dict[str, Any], avg_speed_kmph, experience_months, rider_order_count, rider_avg_delivery):
    """
    Predict rider rating (expects models['rider'] to be a pipeline accepting a DataFrame).
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for predict_rider_rating().") from e

    rider_model = models.get("rider") if isinstance(models, dict) else None
    if rider_model is None:
        raise RuntimeError("Rider model not loaded (models['rider'] is None).")

    df = pd.DataFrame([{
        "avg_speed_kmph": float(avg_speed_kmph),
        "experience_months": float(experience_months),
        "rider_order_count": float(rider_order_count),
        "rider_avg_delivery": float(rider_avg_delivery),
    }])
    pred = rider_model.predict(df)
    return float(pred[0])


def predict_sentiment(models: Dict[str, Any], text: str):
    """
    Predict sentiment. Expects models['sentiment'] to be a pipeline that accepts raw text.
    Returns (int_label, proba_or_None).
    """
    sentiment_model = models.get("sentiment") if isinstance(models, dict) else None
    if sentiment_model is None:
        raise RuntimeError("Sentiment model not loaded (models['sentiment'] is None).")

    pred = sentiment_model.predict([text])[0]
    proba = None
    if hasattr(sentiment_model, "predict_proba"):
        proba = float(sentiment_model.predict_proba([text])[0].max())
    return int(pred), proba


# small helper to expose where data/models should live
def get_paths() -> Dict[str, str]:
    return {"data_dir": DATA_DIR, "model_dir": MODEL_DIR, "chart_dir": CHART_DIR}
