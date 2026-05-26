import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("aqi_model.h5")
scaler = joblib.load("scaler.pkl")

# Page settings
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide"
)

# Title
st.markdown(
    "<h1 style='text-align:center; color:#00FFAA;'>🌍 Air Quality Index Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<center>Deep Learning Based AQI Prediction Using ANN</center>",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("📊 AQI Categories")

st.sidebar.info("""
🟢 Good : 0-50

🟡 Satisfactory : 51-100

🟠 Moderate : 101-200

🔴 Poor : 201-300

🟣 Very Poor : 301-400

⚫ Severe : 401-500
""")

# Layout
col1, col2 = st.columns(2)

# Inputs
with col1:

    st.subheader("Enter Pollution Values")

    pm25 = st.slider("PM2.5", 0.0, 500.0, 120.0)

    pm10 = st.slider("PM10", 0.0, 500.0, 200.0)

    no2 = st.slider("NO2", 0.0, 500.0, 40.0)

    so2 = st.slider("SO2", 0.0, 500.0, 20.0)

    co = st.slider("CO", 0.0, 50.0, 2.0)

    predict = st.button("🚀 Predict AQI")

# Graph
with col2:

    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']

    values = [pm25, pm10, no2, so2, co]

    fig = px.bar(
        x=pollutants,
        y=values,
        title="Pollution Levels"
    )

    st.plotly_chart(fig, use_container_width=True)

# Prediction
if predict:

    # Input array
    data = np.array([[pm25, pm10, no2, so2, co]])

    # Scale input
    data = scaler.transform(data)

    # Predict
    prediction = model.predict(data)

    aqi = float(prediction[0][0])

    # AQI category
    if aqi <= 50:
        category = "Good"
        health = "Safe for outdoor activities"

    elif aqi <= 100:
        category = "Satisfactory"
        health = "Air quality acceptable"

    elif aqi <= 200:
        category = "Moderate"
        health = "Sensitive people take care"

    elif aqi <= 300:
        category = "Poor"
        health = "Wear mask outside"

    elif aqi <= 400:
        category = "Very Poor"
        health = "Avoid outdoor activities"

    else:
        category = "Severe"
        health = "Stay indoors"

    # Metrics
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Predicted AQI", f"{aqi:.2f}")

    with c2:
        st.metric("Category", category)

    with c3:
        st.metric("Health Risk", health)

    # Gauge chart
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        title={'text': "AQI Meter"},
        gauge={
            'axis': {'range': [0, 500]},
            'bar': {'color': "cyan"},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [51, 100], 'color': "yellow"},
                {'range': [101, 200], 'color': "orange"},
                {'range': [201, 300], 'color': "red"},
                {'range': [301, 400], 'color': "purple"},
                {'range': [401, 500], 'color': "black"},
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

# Footer
st.markdown("---")

st.markdown(
    "<center>Made with ❤️ using Deep Learning & Streamlit</center>",
    unsafe_allow_html=True
)