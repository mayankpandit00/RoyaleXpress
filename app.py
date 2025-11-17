# app.py
import streamlit as st
import importlib
import os
import sys

st.set_page_config(page_title='RoyaleXpress', layout='wide')

# -------------------------
# Try to import ml_utils (best-effort)
# -------------------------
_ml_import_error = None
try:
    ml_utils = importlib.import_module("ml_utils")
except Exception as e:
    ml_utils = None
    _ml_import_error = e

# -------------------------
# Local lightweight package check (used if ml_utils isn't available
# or doesn't implement check_requirements)
# -------------------------
def local_check_requirements():
    """Return list of missing package names for ML operations."""
    required = ["joblib", "sklearn", "pandas", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    return missing

def check_requirements():
    """
    Prefer ml_utils.check_requirements() if available, else do local checks.
    Returns a list of missing package names.
    """
    if ml_utils is not None and hasattr(ml_utils, "check_requirements"):
        try:
            return ml_utils.check_requirements()
        except Exception:
            # fallback to local check if ml_utils.check_requirements fails for some reason
            return local_check_requirements()
    else:
        return local_check_requirements()

# -------------------------
# Data loading: attempt to use ml_utils.load_data() if present,
# otherwise try to read CSVs using pandas (if available).
# -------------------------
@st.cache_data(show_spinner=False)
def get_data():
    """
    Returns tuple: users, products, orders, riders, feedback
    If data cannot be loaded, returns (None, None, None, None, None)
    """
    missing = check_requirements()
    # prefer ml_utils.load_data if available
    if ml_utils is not None and hasattr(ml_utils, "load_data"):
        try:
            return ml_utils.load_data()
        except Exception as e:
            st.warning(f"ml_utils.load_data() raised: {e}. Falling back to CSV read if available.")

    # fallback: try to read CSVs from repo root using pandas
    try:
        import pandas as pd
    except Exception as e:
        st.error("Pandas not available; cannot load datasets. Add 'pandas' to requirements.txt to enable data display.")
        return (None, None, None, None, None)

    # Define default CSV filenames (adjust if your files are elsewhere)
    def load_csv_if_exists(fn):
        if os.path.exists(fn):
            try:
                return pd.read_csv(fn)
            except Exception as e:
                st.warning(f"Failed to read {fn}: {e}")
                return None
        return None

    users = load_csv_if_exists("users.csv") or load_csv_if_exists("data/users.csv")
    products = load_csv_if_exists("products.csv") or load_csv_if_exists("data/products.csv")
    orders = load_csv_if_exists("orders.csv") or load_csv_if_exists("data/orders.csv")
    riders = load_csv_if_exists("riders.csv") or load_csv_if_exists("data/riders.csv")
    feedback = load_csv_if_exists("feedback.csv") or load_csv_if_exists("data/feedback.csv")

    # If any are None, create sensible empty DataFrames so basic UI does not crash
    if users is None: users = pd.DataFrame()
    if products is None: products = pd.DataFrame()
    if orders is None: orders = pd.DataFrame()
    if riders is None: riders = pd.DataFrame()
    if feedback is None: feedback = pd.DataFrame()

    return (users, products, orders, riders, feedback)

# -------------------------
# Model loading: only attempt when ML packages are present
# -------------------------
@st.cache_resource(show_spinner=False)
def get_models():
    """
    Safely load models (returns dict of models or None).
    Does not attempt to import joblib/sklearn if they are missing.
    """
    missing = check_requirements()
    # require at least joblib & sklearn to attempt model loads
    required_for_models = {"joblib", "sklearn", "numpy"}
    if any(pkg in missing for pkg in required_for_models):
        return None

    # If ml_utils provides a load_models() use it (preferred)
    if ml_utils is not None and hasattr(ml_utils, "load_models"):
        try:
            return ml_utils.load_models()
        except Exception as e:
            # fallback to manual joblib loads if ml_utils.load_models fails
            st.warning(f"ml_utils.load_models() failed with: {e}. Attempting manual load of joblib models.")
    # Manual fallback using joblib (loads models from 'models' folder by name)
    try:
        import joblib
    except Exception as e:
        st.warning(f"joblib not importable: {e}")
        return None

    model_dir = "models"
    if not os.path.isdir(model_dir):
        st.warning("Model folder 'models' not found in repo. ML features disabled until models are available.")
        return None

    # Example model filenames - adjust to your actual filenames
    model_map = {
        "eta": os.path.join(model_dir, "eta.joblib"),
        "highvalue": os.path.join(model_dir, "highvalue.joblib"),
        "cluster": os.path.join(model_dir, "kmeans_users.joblib"),
        "scaler": os.path.join(model_dir, "kmeans_scaler.joblib"),
        "rating": os.path.join(model_dir, "rating.joblib"),
        "sentiment_clf": os.path.join(model_dir, "sentiment_clf.joblib"),
        "sentiment_vect": os.path.join(model_dir, "sentiment_vect.joblib"),
    }
    loaded = {}
    for key, path in model_map.items():
        if os.path.exists(path):
            try:
                loaded[key] = joblib.load(path)
            except Exception as e:
                st.warning(f"Failed loading model {path}: {e}")
        else:
            # file missing - that's fine; we load what exists
            loaded[key] = None
    return loaded

# -------------------------
# Utility wrappers for predictions that guard against missing models/deps
# -------------------------
def safe_predict_eta(models, *args, **kwargs):
    if models is None:
        st.warning("ETA prediction unavailable: models or ML packages missing.")
        return None
    # Prefer ml_utils.predict_eta if available
    if ml_utils is not None and hasattr(ml_utils, "predict_eta"):
        try:
            return ml_utils.predict_eta(models, *args, **kwargs)
        except Exception as e:
            st.error(f"ml_utils.predict_eta() failed: {e}")
            return None
    # Manual fallback: if user supplied an 'eta' model object that has predict
    eta_model = models.get("eta") if isinstance(models, dict) else None
    if eta_model is None:
        st.warning("No ETA model found in loaded models.")
        return None
    try:
        # assume model expects a 2D array-like input; user should adapt as per their pipeline
        import numpy as np
        X = np.array([args])
        return float(eta_model.predict(X)[0])
    except Exception as e:
        st.error(f"Fallback ETA prediction failed: {e}")
        return None

def safe_predict_highvalue(models, *args, **kwargs):
    if models is None:
        st.warning("High-value prediction unavailable: models or ML packages missing.")
        return None, None
    if ml_utils is not None and hasattr(ml_utils, "predict_highvalue"):
        try:
            return ml_utils.predict_highvalue(models, *args, **kwargs)
        except Exception as e:
            st.error(f"ml_utils.predict_highvalue() failed: {e}")
            return None, None
    # fallback if model present
    hv_model = models.get("highvalue") if isinstance(models, dict) else None
    if hv_model is None:
        st.warning("No High-Value model found in loaded models.")
        return None, None
    try:
        import numpy as np
        X = np.array([args])
        pred = hv_model.predict(X)[0]
        proba = None
        if hasattr(hv_model, "predict_proba"):
            proba = float(hv_model.predict_proba(X)[0, 1])
        return int(pred), proba
    except Exception as e:
        st.error(f"Fallback high-value prediction failed: {e}")
        return None, None

def safe_predict_rider_rating(models, *args, **kwargs):
    if models is None:
        st.warning("Rider rating prediction unavailable: models or ML packages missing.")
        return None
    if ml_utils is not None and hasattr(ml_utils, "predict_rider_rating"):
        try:
            return ml_utils.predict_rider_rating(models, *args, **kwargs)
        except Exception as e:
            st.error(f"ml_utils.predict_rider_rating() failed: {e}")
            return None
    rating_model = models.get("rating") if isinstance(models, dict) else None
    if rating_model is None:
        st.warning("No Rider Rating model found in loaded models.")
        return None
    try:
        import numpy as np
        X = np.array([args])
        return float(rating_model.predict(X)[0])
    except Exception as e:
        st.error(f"Fallback rider rating prediction failed: {e}")
        return None

def safe_predict_sentiment(models, text):
    if models is None:
        st.warning("Sentiment analysis unavailable: models or ML packages missing.")
        return None, None
    if ml_utils is not None and hasattr(ml_utils, "predict_sentiment"):
        try:
            return ml_utils.predict_sentiment(models, text)
        except Exception as e:
            st.error(f"ml_utils.predict_sentiment() failed: {e}")
            return None, None
    # fallback: try using vectorizer & clf if present in models
    vect = models.get("sentiment_vect") if isinstance(models, dict) else None
    clf = models.get("sentiment_clf") if isinstance(models, dict) else None
    if vect is None or clf is None:
        st.warning("No sentiment vectorizer/classifier found in loaded models.")
        return None, None
    try:
        X = vect.transform([text])
        pred = int(clf.predict(X)[0])
        proba = None
        if hasattr(clf, "predict_proba"):
            proba = float(clf.predict_proba(X)[0].max())
        return pred, proba
    except Exception as e:
        st.error(f"Fallback sentiment prediction failed: {e}")
        return None, None

# -------------------------
# Prepare datasets & models (safe)
# -------------------------
users, products, orders, riders, feedback = get_data()
models = get_models()
missing = check_requirements()

# -------------------------
# Build UI
# -------------------------
tabs = st.tabs(["Home","ETA","High-Value","Segments","Riders","Sentiment","Charts"])

# --- HOME tab ---
with tabs[0]:
    st.title("RoyaleXpress - Luxury Instant Delivery")
    if users is None or orders is None or feedback is None:
        st.warning("One or more datasets could not be loaded. Check logs and ensure CSVs or ml_utils.load_data() are available.")
    # metrics (guarded)
    try:
        col1, col2, col3, col4 = st.columns(4)
        total_orders = len(orders)
        avg_delivery = orders['delivery_time_minutes'].mean() if 'delivery_time_minutes' in orders.columns else float('nan')
        revenue = orders['order_value'].sum() if 'order_value' in orders.columns else 0
        pos_pct = (feedback['rating']>=4).mean()*100 if 'rating' in feedback.columns else 0.0

        col1.metric("Total Orders", f"{total_orders}")
        col2.metric("Avg Delivery (min)", f"{avg_delivery:.1f}" if not (avg_delivery!=avg_delivery) else "N/A")
        col3.metric("Total Revenue", f"₹{int(revenue):,}")
        col4.metric("Positive Feedback %", f"{pos_pct:.1f}%")
    except Exception as e:
        st.warning(f"Couldn't compute some metrics: {e}")

    st.markdown("---")
    st.subheader("Datasets Preview")
    if hasattr(users, "head"):
        st.dataframe(users.head(10))
    else:
        st.write("Users dataset not available to preview.")

# --- ETA tab ---
with tabs[1]:
    st.header("ETA Prediction")
    if models is None or any(pkg in missing for pkg in ("joblib","sklearn")):
        st.warning("ML dependencies are missing or models not loaded. ETA predictions disabled. Missing: " + ", ".join(missing))
    st.write("Provide order/rider info to get predicted delivery time.")
    c1, c2 = st.columns(2)
    with c1:
        distance_km = st.number_input('Distance (km)', min_value=0.0, value=2.5)
        traffic_level = st.selectbox('Traffic level', ['low','moderate','high'])
        order_value = st.number_input('Order value (₹)', min_value=0, value=5000)
    with c2:
        avg_speed = st.number_input('Rider avg speed (kmph)', min_value=5.0, value=18.0)
        loyalty_score = st.number_input('User loyalty score (1-100)', min_value=1, max_value=100, value=60)
        if st.button('Predict ETA'):
            pred = safe_predict_eta(models, distance_km, traffic_level, avg_speed, loyalty_score, order_value)
            if pred is not None:
                st.success(f"Predicted delivery time: {pred:.1f} minutes")
                chart_path = os.path.join('charts','eta_true_vs_pred.png')
                if os.path.exists(chart_path):
                    st.image(chart_path)
                else:
                    st.info("No ETA chart file found to display.")
            else:
                st.info("ETA prediction not available.")

# --- High-Value tab ---
with tabs[2]:
    st.header("High-Value Customer Prediction")
    if models is None:
        st.warning("High-value prediction disabled (models/packages missing).")
    st.write("Input user-level features to predict whether the user is high-value.")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        loyalty = st.number_input('Loyalty score', min_value=1, max_value=100, value=60)
    with col2:
        order_count = st.number_input('Order count', min_value=0, value=5)
        avg_order_value = st.number_input('Avg order value (₹)', min_value=0, value=4500)
    with col3:
        avg_delivery_time = st.number_input('Avg delivery time (min)', min_value=1.0, value=20.0)
    if st.button('Predict High-Value'):
        pred, proba = safe_predict_highvalue(models, age, loyalty, order_count, avg_order_value, avg_delivery_time)
        if pred is not None:
            st.write('Prediction (1 = high value, 0 = not):', pred)
            if proba is not None:
                st.write('Probability:', f"{proba:.2f}")
            chart_path = os.path.join('charts', 'highvalue_feature_importance.png')
            if os.path.exists(chart_path):
                st.image(chart_path)
        else:
            st.info("High-value prediction not available.")

# --- Segments tab ---
with tabs[3]:
    st.header("Customer Segments Explorer")
    st.write("Segments are precomputed using KMeans (4 clusters). Select a cluster to inspect users.")
    seg_df = None
    try:
        if users is not None and hasattr(users, "copy"):
            if 'cluster' not in users.columns:
                st.info('Cluster column not found in users dataset. Attempting to compute clusters if scaler+kmeans models exist.')
                # Only attempt to compute clusters if scaler + kmeans model present and sklearn available
                if models is None or models.get("cluster") is None or models.get("scaler") is None:
                    st.warning("KMeans/scaler models missing; showing users without cluster assignment.")
                    seg_df = users.copy()
                else:
                    # use loaded scaler and kmeans
                    scaler = models.get("scaler")
                    kmeans = models.get("cluster")
                    try:
                        seg_df = users.copy()
                        seg_df['order_count'] = seg_df.get('order_count', 0)
                        arr = scaler.transform(seg_df[['age','avg_monthly_spend','loyalty_score','order_count']])
                        seg_df['cluster'] = kmeans.predict(arr)
                    except Exception as e:
                        st.warning(f"Failed to compute clusters: {e}")
                        seg_df = users.copy()
            else:
                seg_df = users.copy()
        else:
            st.warning("Users dataset not available.")
            seg_df = None
    except Exception as e:
        st.warning(f"Error preparing segments: {e}")
        seg_df = users.copy() if users is not None else None

    if seg_df is not None and 'cluster' in seg_df.columns:
        sel = st.selectbox('Select cluster', sorted(seg_df['cluster'].unique().tolist()))
        st.write('Cluster size:', int((seg_df['cluster']==sel).sum()))
        st.dataframe(seg_df[seg_df['cluster']==sel].head(20))
    else:
        st.write("No cluster information available to display.")

    chart_path = os.path.join('charts','kmeans_user_segments.png')
    if os.path.exists(chart_path):
        st.image(chart_path)

# --- Riders tab ---
with tabs[4]:
    st.header('Rider Performance Predictor')
    st.write('Predict expected rider rating given rider stats.')
    c1, c2 = st.columns(2)
    with c1:
        avg_speed_r = st.number_input('Avg speed (kmph)', min_value=5.0, value=20.0, key="r_avg_speed")
        experience = st.number_input('Experience (months)', min_value=0, value=12, key="r_experience")
    with c2:
        rider_order_count = st.number_input('Order count (rider)', min_value=0, value=50, key="r_order_count")
        rider_avg_delivery = st.number_input('Avg delivery time (min)', min_value=1.0, value=18.0, key="r_avg_delivery")
    if st.button('Predict Rider Rating'):
        pred = safe_predict_rider_rating(models, avg_speed_r, experience, rider_order_count, rider_avg_delivery)
        if pred is not None:
            st.success(f'Predicted rider rating: {pred:.2f} / 5')
            chart_path = os.path.join('charts','rider_feature_importance.png')
            if os.path.exists(chart_path):
                st.image(chart_path)
        else:
            st.info("Rider rating prediction not available.")

# --- Sentiment tab ---
with tabs[5]:
    st.header('Sentiment Analysis')
    text = st.text_area('Paste a review or feedback text here', value='Fast delivery, excellent packaging.')
    if st.button('Analyze Sentiment'):
        pred, proba = safe_predict_sentiment(models, text)
        if pred is not None:
            lbl = 'Positive' if pred == 1 else 'Negative'
            st.write('Sentiment:', lbl)
            if proba is not None:
                st.write('Confidence:', f"{proba:.2f}")
            chart_path = os.path.join('charts','sentiment_distribution.png')
            if os.path.exists(chart_path):
                st.image(chart_path)
        else:
            st.info("Sentiment analysis not available.")

# --- Charts tab ---
with tabs[6]:
    st.header('Saved Charts')
    st.write('Pre-generated charts from training. You can also upload new charts to display.')
    charts_dir = "charts"
    if os.path.isdir(charts_dir):
        chart_files = [f for f in os.listdir(charts_dir) if f.lower().endswith('.png')]
        if len(chart_files) == 0:
            st.info("No chart PNGs found in the 'charts' folder.")
        for c in chart_files:
            st.subheader(c)
            st.image(os.path.join(charts_dir, c))
    else:
        st.info("No 'charts' folder found in the repository.")

# -------------------------
# Final notes (visible in UI)
# -------------------------
with st.expander("Deployment / Debug info"):
    st.write("ML module import error (if any):")
    if _ml_import_error:
        st.write(str(_ml_import_error))
    else:
        st.write("No import error for ml_utils module.")
    st.write("Missing packages detected:")
    st.write(", ".join(check_requirements()) or "None")
    st.write("Loaded model keys (if any):")
    try:
        st.write(list(models.keys()) if isinstance(models, dict) else ("None" if models is None else str(type(models))))
    except Exception:
        st.write("Models not available.")
