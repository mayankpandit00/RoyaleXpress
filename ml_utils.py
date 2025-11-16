import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "charts"

MODEL_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

def _download_url_to(path: Path, url: str):
    import requests
    path_tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path_tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    path_tmp.rename(path)

def _ensure_model(name: str, url_secret_key: str = None):
    p = MODEL_DIR / name
    if p.exists():
        return p
    try:
        import streamlit as st
    except Exception:
        st = None
    if st and url_secret_key and url_secret_key in st.secrets:
        url = st.secrets[url_secret_key]
        _download_url_to(p, url)
        return p
    raise FileNotFoundError(f"Model file not found: {p}. Commit it to models/ or set st.secrets['{url_secret_key}']='https://...'")

def load_data():
    """Load CSVs from ./data folder (lazy import)."""
    import pandas as pd
    users = pd.read_csv(DATA_DIR / "users.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    riders = pd.read_csv(DATA_DIR / "riders.csv")
    feedback = pd.read_csv(DATA_DIR / "feedback.csv")
    return users, products, orders, riders, feedback

def load_models():
    """
    Lazy-load model objects. Raises a clear error if joblib is missing.
    """
    try:
        import joblib
    except Exception as e:
        raise RuntimeError(
            "Required dependency 'joblib' is missing in the environment. "
            "Add 'joblib' to requirements.txt and push to GitHub so Streamlit Cloud installs it."
        ) from e

    mapping = {
        "eta_gbr_pipeline.joblib": "MODEL_ETA_URL",
        "highvalue_rf_pipeline.joblib": "MODEL_HV_URL",
        "kmeans_users.joblib": "MODEL_KMEANS_URL",
        "kmeans_scaler.joblib": "MODEL_KMEANS_SCALER_URL",
        "riderperf_rf_pipeline.joblib": "MODEL_RIDER_URL",
        "sentiment_tfidf_lr.joblib": "MODEL_SENT_URL",
    }

    models = {}
    for fname, secret_key in mapping.items():
        model_path = _ensure_model(fname, secret_key)
        models[fname.split(".joblib")[0]] = joblib.load(model_path)

    loaded = {
        "eta": models.get("eta_gbr_pipeline"),
        "highvalue": models.get("highvalue_rf_pipeline"),
        "kmeans": models.get("kmeans_users"),
        "kmeans_scaler": models.get("kmeans_scaler"),
        "rider": models.get("riderperf_rf_pipeline"),
        "sentiment": models.get("sentiment_tfidf_lr"),
    }
    return loaded

# Prediction wrappers (lazy imports)
def predict_eta(models, distance_km, traffic_level, avg_speed_kmph, loyalty_score, order_value):
    import pandas as pd
    df = pd.DataFrame([{
        'distance_km': float(distance_km),
        'traffic_level': str(traffic_level),
        'avg_speed_kmph': float(avg_speed_kmph),
        'loyalty_score': float(loyalty_score),
        'order_value': float(order_value)
    }])
    pred = models['eta'].predict(df)
    return float(pred[0])

def predict_highvalue(models, age, loyalty_score, order_count, avg_order_value, avg_delivery_time):
    import pandas as pd
    df = pd.DataFrame([{
        'age': float(age),
        'loyalty_score': float(loyalty_score),
        'order_count': float(order_count),
        'avg_order_value': float(avg_order_value),
        'avg_delivery_time': float(avg_delivery_time)
    }])
    pred = models['highvalue'].predict(df)
    proba = None
    if hasattr(models['highvalue'], "predict_proba"):
        proba = float(models['highvalue'].predict_proba(df)[0,1])
    return int(pred[0]), proba

def assign_user_cluster(models, user_row):
    import numpy as np
    arr = np.array([[user_row['age'], user_row['avg_monthly_spend'], user_row['loyalty_score'], user_row['order_count']]])
    scaled = models['kmeans_scaler'].transform(arr)
    cluster = int(models['kmeans'].predict(scaled)[0])
    return cluster

def predict_rider_rating(models, avg_speed_kmph, experience_months, rider_order_count, rider_avg_delivery):
    import pandas as pd
    df = pd.DataFrame([{
        'avg_speed_kmph': float(avg_speed_kmph),
        'experience_months': float(experience_months),
        'rider_order_count': float(rider_order_count),
        'rider_avg_delivery': float(rider_avg_delivery)
    }])
    pred = models['rider'].predict(df)
    return float(pred[0])

def predict_sentiment(models, text):
    pred = models['sentiment'].predict([text])[0]
    proba = None
    if hasattr(models['sentiment'], "predict_proba"):
        proba = float(models['sentiment'].predict_proba([text])[0].max())
    return int(pred), proba


