# ml_utils.py
"""
Lightweight ML utilities module.
All heavy deps (joblib, sklearn, pandas, etc.) are imported inside functions
so importing this module does not require those packages to be installed.
"""

def check_requirements():
    """
    Return a list of missing packages required for ML operations.
    Use this from Streamlit to show a helpful message instead of crashing.
    """
    required = ("joblib", "sklearn", "pandas", "numpy")
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def load_data(path):
    """
    Load data from `path`. This import is local so Streamlit UI can import ml_utils
    even if pandas isn't installed.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for load_data(). Add 'pandas' to requirements.txt.") from e

    # Example: adjust to your actual loading code
    df = pd.read_csv(path)
    return df


def load_models(model_paths):
    """
    Load one or more joblib models. model_paths can be a single path or dict/list.
    """
    try:
        import joblib
    except ImportError as e:
        raise ImportError("joblib is required for load_models(). Add 'joblib' to requirements.txt.") from e

    # adapt to your file structure; example for dict of models:
    if isinstance(model_paths, dict):
        models = {}
        for name, p in model_paths.items():
            models[name] = joblib.load(p)
        return models
    else:
        return joblib.load(model_paths)


def predict_eta(model, X):
    """
    Example predict function. Import sklearn/numpy locally if needed.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for predict_eta(). Add 'numpy' to requirements.txt.") from e

    # your real preprocessing/prediction logic here
    return model.predict(X)


def predict_highvalue(model, X):
    # same pattern: import libs locally if needed
    return model.predict(X)


def assign_user_cluster(clustering_model, X):
    return clustering_model.predict(X)


def predict_rider_rating(model, X):
    return model.predict(X)


def predict_sentiment(text, vectorizer=None, clf=None):
    """
    If you use a vectorizer and classifier, pass them in. Imports done locally.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for predict_sentiment(). Add 'numpy' to requirements.txt.") from e

    if vectorizer is None or clf is None:
        raise ValueError("vectorizer and clf must be provided to predict_sentiment().")
    X = vectorizer.transform([text])
    return clf.predict(X)[0]
