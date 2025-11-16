import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

RSEED = 42
np.random.seed(RSEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

print("Loading data from", DATA_DIR)
users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
riders = pd.read_csv(os.path.join(DATA_DIR, "riders.csv"))
orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
feedback = pd.read_csv(os.path.join(DATA_DIR, "feedback.csv"))

# ---------- 1) ETA regression ----------
print("Training ETA model...")
orders_eta = orders.merge(riders[['rider_id','avg_speed_kmph','rating','experience_months']],
                          left_on='delivery_partner_id', right_on='rider_id', how='left')
orders_eta = orders_eta.merge(users[['user_id','loyalty_score']], on='user_id', how='left')
X_eta = orders_eta[['distance_km','traffic_level','avg_speed_kmph','loyalty_score','order_value']].copy()
y_eta = orders_eta['delivery_time_minutes'].astype(float)

num_feats = ['distance_km','avg_speed_kmph','loyalty_score','order_value']
cat_feats = ['traffic_level']

num_pipeline = Pipeline([('scaler', StandardScaler())])
cat_pipeline = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
preproc_eta = ColumnTransformer([('num', num_pipeline, num_feats),
                                 ('cat', cat_pipeline, cat_feats)])
model_eta = Pipeline([('prep', preproc_eta),
                      ('gb', GradientBoostingRegressor(random_state=RSEED))])

X_tr, X_te, y_tr, y_te = train_test_split(X_eta, y_eta, test_size=0.2, random_state=RSEED)
model_eta.fit(X_tr, y_tr)
preds = model_eta.predict(X_te)
print("ETA MAE:", mean_absolute_error(y_te, preds), "R2:", r2_score(y_te, preds))
joblib.dump(model_eta, os.path.join(MODEL_DIR, "eta_gbr_pipeline.joblib"))

plt.figure(figsize=(6,4))
plt.scatter(y_te, preds, alpha=0.6)
plt.xlabel("True delivery_time_minutes"); plt.ylabel("Predicted delivery_time_minutes")
plt.title("ETA: True vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "eta_true_vs_pred.png"))
plt.close()

# ---------- 2) High-value classifier ----------
print("Training High-value classifier...")
user_orders = orders.groupby('user_id').agg(order_count=('order_id','count'),
                                            avg_order_value=('order_value','mean'),
                                            avg_delivery_time=('delivery_time_minutes','mean')).reset_index()
users_ml = users.merge(user_orders, on='user_id', how='left').fillna(0)
X_hv = users_ml[['age','loyalty_score','order_count','avg_order_value','avg_delivery_time']].copy()
y_hv = users_ml['high_value_flag']

prep_hv = ColumnTransformer([('num', StandardScaler(), X_hv.columns.tolist())])
model_hv = Pipeline([('prep', prep_hv),
                     ('rf', RandomForestClassifier(n_estimators=200, random_state=RSEED))])

X_tr, X_te, y_tr, y_te = train_test_split(X_hv, y_hv, test_size=0.2, random_state=RSEED, stratify=y_hv)
model_hv.fit(X_tr, y_tr)
preds = model_hv.predict(X_te)
print("High-value Accuracy:", accuracy_score(y_te, preds), "F1:", f1_score(y_te, preds, zero_division=0))
print(classification_report(y_te, preds))
joblib.dump(model_hv, os.path.join(MODEL_DIR, "highvalue_rf_pipeline.joblib"))

# Feature importance chart
rf = model_hv.named_steps['rf']
importances = rf.feature_importances_
plt.figure(figsize=(6,4))
plt.bar(X_hv.columns.tolist(), importances)
plt.title("High-value Feature Importances"); plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "highvalue_feature_importance.png"))
plt.close()

# ---------- 3) KMeans user segmentation ----------
print("Training KMeans for segmentation...")
seg_feats = users_ml[['age','avg_monthly_spend','loyalty_score','order_count']].copy()
scaler_seg = StandardScaler()
seg_scaled = scaler_seg.fit_transform(seg_feats)
kmeans = KMeans(n_clusters=4, random_state=RSEED)
clusters = kmeans.fit_predict(seg_scaled)
users_ml['cluster'] = clusters
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_users.joblib"))
joblib.dump(scaler_seg, os.path.join(MODEL_DIR, "kmeans_scaler.joblib"))

from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=RSEED)
proj = pca.fit_transform(seg_scaled)
plt.figure(figsize=(6,4))
for c in np.unique(clusters):
    mask = clusters==c
    plt.scatter(proj[mask,0], proj[mask,1], label=f"cluster {c}", alpha=0.6)
plt.legend(); plt.title("User Segments (KMeans)"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig(os.path.join(CHART_DIR, "kmeans_user_segments.png")); plt.close()

# ---------- 4) Rider performance regressor ----------
print("Training Rider performance regressor...")
rider_orders = orders.groupby('delivery_partner_id').agg(rider_order_count=('order_id','count'),
                                                        rider_avg_delivery=('delivery_time_minutes','mean')).reset_index().rename(columns={'delivery_partner_id':'rider_id'})
rider_ml = riders.merge(rider_orders, on='rider_id', how='left').fillna(0)
X_rider = rider_ml[['avg_speed_kmph','experience_months','rider_order_count','rider_avg_delivery']].copy()
y_rider = rider_ml['rating']

prep_rider = ColumnTransformer([('num', StandardScaler(), X_rider.columns.tolist())])
model_rider = Pipeline([('prep', prep_rider),
                        ('rf', RandomForestRegressor(n_estimators=200, random_state=RSEED))])

X_tr, X_te, y_tr, y_te = train_test_split(X_rider, y_rider, test_size=0.2, random_state=RSEED)
model_rider.fit(X_tr, y_tr)
preds = model_rider.predict(X_te)
print("Rider MAE:", mean_absolute_error(y_te, preds), "R2:", r2_score(y_te, preds))
joblib.dump(model_rider, os.path.join(MODEL_DIR, "riderperf_rf_pipeline.joblib"))

plt.figure(figsize=(6,4))
importances = model_rider.named_steps['rf'].feature_importances_
plt.bar(X_rider.columns.tolist(), importances)
plt.title("Rider Feature Importances"); plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig(os.path.join(CHART_DIR, "rider_feature_importance.png"))
plt.close()

# ---------- 5) Sentiment TF-IDF + LR ----------
print("Training Sentiment model...")
feedback['sentiment'] = (feedback['rating'] >= 4.0).astype(int)
X_text = feedback['review'].fillna("")
y_sent = feedback['sentiment']

X_tr, X_te, y_tr, y_te = train_test_split(X_text, y_sent, test_size=0.2, random_state=RSEED, stratify=y_sent)
tfidf = TfidfVectorizer(max_features=500)
clf = Pipeline([('tfidf', tfidf), ('lr', LogisticRegression(max_iter=1000))])
clf.fit(X_tr, y_tr)
preds = clf.predict(X_te)
print("Sentiment Accuracy:", accuracy_score(y_te, preds), "F1:", f1_score(y_te, preds, zero_division=0))
joblib.dump(clf, os.path.join(MODEL_DIR, "sentiment_tfidf_lr.joblib"))

plt.figure(figsize=(4,3))
sent_counts = feedback['sentiment'].value_counts().sort_index()
plt.bar(['negative','positive'], sent_counts.values)
plt.title("Sentiment Distribution"); plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "sentiment_distribution.png"))
plt.close()

print("Retraining complete. Models saved to", MODEL_DIR)
