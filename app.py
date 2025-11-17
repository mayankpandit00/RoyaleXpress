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
    # Manual fallback using joblib (loads models from 'mo
