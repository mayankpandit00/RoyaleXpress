import streamlit as st
import pandas as pd
import os
from ml_utils import load_data, load_models, predict_eta, predict_highvalue, assign_user_cluster, predict_rider_rating, predict_sentiment
import joblib
import os

# Basic config
st.set_page_config(page_title='RoyaleXpress', layout='wide')


# Caching load operations
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_models():
    return load_models()


users, products, orders, riders, feedback = get_data()
models = get_models()

# Top-level tabs
tabs = st.tabs(["Home","ETA","High-Value","Segments","Riders","Sentiment","Charts"])

# --- HOME tab ---
with tabs[0]:
    st.title("RoyaleXpress - Luxury Instant Delivery")
    col1, col2, col3, col4 = st.columns(4)
    total_orders = len(orders)
    avg_delivery = orders['delivery_time_minutes'].mean()
    revenue = orders['order_value'].sum()
    pos_pct = (feedback['rating']>=4).mean()*100


    col1.metric("Total Orders", f"{total_orders}")
    col2.metric("Avg Delivery (min)", f"{avg_delivery:.1f}")
    col3.metric("Total Revenue", f"₹{int(revenue):,}")
    col4.metric("Positive Feedback %", f"{pos_pct:.1f}%")


    st.markdown("---")
    st.subheader("Datasets Preview")
    st.dataframe(users.head(10))
    st.write("\n")

# --- ETA tab ---
with tabs[1]:
    st.header("ETA Prediction")
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
            pred = predict_eta(models, distance_km, traffic_level, avg_speed, loyalty_score, order_value)
            st.success(f"Predicted delivery time: {pred:.1f} minutes")
            st.image(os.path.join('charts','eta_true_vs_pred.png'))

# --- High-Value tab ---
with tabs[2]:
    st.header("High-Value Customer Prediction")
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
        pred, proba = predict_highvalue(
            models,
            age,
            loyalty,
            order_count,
            avg_order_value,
            avg_delivery_time
        )
        st.write('Prediction (1 = high value, 0 = not):', pred)
        if proba is not None:
            st.write('Probability:', f"{proba:.2f}")
        st.image(os.path.join('charts', 'highvalue_feature_importance.png'))


# --- Segments tab ---
with tabs[3]:
    st.header("Customer Segments Explorer")
    st.write("Segments are precomputed using KMeans (4 clusters). Select a cluster to inspect users.")
    if 'cluster' not in users.columns:
        # load cluster assignment from orders/users join performed earlier in the pipeline if available
        # fallback: read a precomputed mapping from a csv if you saved one; otherwise recompute minimal version
        st.info('Cluster column not found in users.csv. Using kmeans to compute clusters on the fly.')
        from sklearn.preprocessing import StandardScaler
        scaler = joblib.load(os.path.join('models','kmeans_scaler.joblib'))
        kmeans = joblib.load(os.path.join('models','kmeans_users.joblib'))
        seg_df = users.copy()
        seg_df['order_count'] = seg_df.get('order_count', 0)
        arr = scaler.transform(seg_df[['age','avg_monthly_spend','loyalty_score','order_count']])
        seg_df['cluster'] = kmeans.predict(arr)
    else:
        seg_df = users.copy()
    sel = st.selectbox('Select cluster', sorted(seg_df['cluster'].unique().tolist()))
    st.write('Cluster size:', int((seg_df['cluster']==sel).sum()))
    st.dataframe(seg_df[seg_df['cluster']==sel].head(20))
    st.image(os.path.join('charts','kmeans_user_segments.png'))

# --- Riders tab ---
with tabs[4]:
    st.header('Rider Performance Predictor')
    st.write('Predict expected rider rating given rider stats.')
    c1, c2 = st.columns(2)
    with c1:
        avg_speed = st.number_input('Avg speed (kmph)', min_value=5.0, value=20.0)
        experience = st.number_input('Experience (months)', min_value=0, value=12)
    with c2:
        rider_order_count = st.number_input('Order count (rider)', min_value=0, value=50)
        rider_avg_delivery = st.number_input('Avg delivery time (min)', min_value=1.0, value=18.0)
    if st.button('Predict Rider Rating'):
        pred = predict_rider_rating(models, avg_speed, experience, rider_order_count, rider_avg_delivery)
        st.success(f'Predicted rider rating: {pred:.2f} / 5')
        st.image(os.path.join('charts','rider_feature_importance.png'))

# --- Sentiment tab ---
with tabs[5]:
    st.header('Sentiment Analysis')
    text = st.text_area('Paste a review or feedback text here', value='Fast delivery, excellent packaging.')
    if st.button('Analyze Sentiment'):
        pred, proba = predict_sentiment(models, text)
        lbl = 'Positive' if pred==1 else 'Negative'
        st.write('Sentiment:', lbl)
        if proba is not None:
            st.write('Confidence:', f"{proba:.2f}")
        st.image(os.path.join('charts','sentiment_distribution.png'))

# --- Charts tab ---
with tabs[6]:
    st.header('Saved Charts')
    st.write('Pre-generated charts from training. You can also upload new charts to display.')
    chart_files = [f for f in os.listdir('charts') if f.endswith('.png')]
    for c in chart_files:
        st.subheader(c)
        st.image(os.path.join('charts', c))
