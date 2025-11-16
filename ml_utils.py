import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')

def load_data():
    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    riders = pd.read_csv(os.path.join(DATA_DIR, 'riders.csv'))
    feedback = pd.read_csv(os.path.join(DATA_DIR, 'feedback.csv'))
    return users, products, orders, riders, feedback

def load_models():
    models = {}
    models['eta'] = joblib.load(os.path.join(MODEL_DIR, 'eta_gbr_pipeline.joblib'))
    models['highvalue'] = joblib.load(os.path.join(MODEL_DIR, 'highvalue_rf_pipeline.joblib'))
    models['kmeans'] = joblib.load(os.path.join(MODEL_DIR, 'kmeans_users.joblib'))
    models['kmeans_scaler'] = joblib.load(os.path.join(MODEL_DIR, 'kmeans_scaler.joblib'))
    models['rider'] = joblib.load(os.path.join(MODEL_DIR, 'riderperf_rf_pipeline.joblib'))
    models['sentiment'] = joblib.load(os.path.join(MODEL_DIR, 'sentiment_tfidf_lr.joblib'))
    return models

def predict_eta(models, distance_km, traffic_level, avg_speed_kmph, loyalty_score, order_value):
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
    df = pd.DataFrame([{
    'age': float(age),
    'loyalty_score': float(loyalty_score),
    'order_count': float(order_count),
    'avg_order_value': float(avg_order_value),
    'avg_delivery_time': float(avg_delivery_time)
    }])
    pred = models['highvalue'].predict(df)
    proba = models['highvalue'].predict_proba(df)[0,1] if hasattr(models['highvalue'], 'predict_proba') else None
    return int(pred[0]), float(proba) if proba is not None else None

def assign_user_cluster(models, user_row):
    arr = np.array([[user_row['age'], user_row['avg_monthly_spend'], user_row['loyalty_score'], user_row['order_count']]])
    scaled = models['kmeans_scaler'].transform(arr)
    cluster = int(models['kmeans'].predict(scaled)[0])
    return cluster

def predict_rider_rating(models, avg_speed_kmph, experience_months, rider_order_count, rider_avg_delivery):
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
    proba = models['sentiment'].predict_proba([text])[0].max() if hasattr(models['sentiment'], 'predict_proba') else None
    return int(pred), float(proba) if proba is not None else None