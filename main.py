import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import plotly.graph_objects as go

# ------------------------------
# Load trained pipeline
# ------------------------------
pipeline = joblib.load(r"D:\data science\projects seriis\polonomial\crop_yield_pipeline.pkl")
data = pd.read_csv(r"D:\data science\projects seriis\polonomial\cropdata\yield_df.csv")

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Hero Header
# ------------------------------
st.markdown("""
<div style="
    text-align:center;
    padding:40px;
    border-radius:15px;
    background: linear-gradient(to right, #2b7a78, #3aafa9);
    color:white;
    font-family: 'Arial', sans-serif;">
    <h1 style='font-size:50px; font-weight:bold;'>🌾 Crop Yield Predictor</h1>
    <p style='font-size:20px; margin-top:10px;'>Predict your crop yield using environmental factors</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("📝 Enter Crop Details")
item = st.sidebar.selectbox("🌾 Select Item", sorted(data["Item"].unique()))
area = st.sidebar.selectbox("📍 Select Area", sorted(data["Area"].unique()))
rainfall = st.sidebar.number_input(
    "☔ Average Rainfall (mm/year)",
    min_value=0.0,
    value=float(data["average_rain_fall_mm_per_year"].mean())
)
pesticides = st.sidebar.number_input(
    "🧪 Pesticides Used (tonnes)",
    min_value=0.0,
    value=float(data["pesticides_tonnes"].mean())
)
temperature = st.sidebar.slider(
    "🌡️ Average Temperature (°C)",
    min_value=-10,
    max_value=50,
    value=int(data["avg_temp"].mean())
)

# ------------------------------


# ------------------------------
# Predict Button
# ------------------------------
if st.sidebar.button("Predict Yield 🌱"):
    input_df = pd.DataFrame({
        "Item": [item],
        "Area": [area],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [temperature]
    })

    pred = pipeline.predict(input_df)
    pred = np.clip(pred, 0, None)

    # ------------------------------
    # Result Card
    # ------------------------------
    st.markdown(f"""
    <div style='text-align:center; padding:30px; border-radius:20px;
                background: linear-gradient(135deg, #3aafa9, #def6f1);
                box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin-top:20px;'>
        <h2 style='color:#17252a;'>🌱 Predicted Crop Yield</h2>
        <h1 style='color:#17252a; font-size:50px; font-weight:bold;'>{pred[0]:.2f} tonnes /area</h1>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------
    

    # ------------------------------
    # Sensitivity Graph (Interactive)
    # ------------------------------
    st.subheader("📊 Yield Sensitivity Analysis")

    # Variation ranges
    rainfall_range = np.linspace(rainfall*0.5, rainfall*1.5, 20)
    temp_range = np.linspace(max(-10, temperature-10), min(50, temperature+10), 20)
    pesticides_range = np.linspace(max(0, pesticides-1), pesticides+1, 20)

    # Predict variations
    df_rain = pd.DataFrame({"Item":[item]*20,"Area":[area]*20,
                            "average_rain_fall_mm_per_year":rainfall_range,
                            "pesticides_tonnes":[pesticides]*20,
                            "avg_temp":[temperature]*20})
    pred_rain = np.clip(pipeline.predict(df_rain),0,None)

    df_temp = pd.DataFrame({"Item":[item]*20,"Area":[area]*20,
                            "average_rain_fall_mm_per_year":[rainfall]*20,
                            "pesticides_tonnes":[pesticides]*20,
                            "avg_temp":temp_range})
    pred_temp = np.clip(pipeline.predict(df_temp),0,None)

    df_pest = pd.DataFrame({"Item":[item]*20,"Area":[area]*20,
                            "average_rain_fall_mm_per_year":[rainfall]*20,
                            "pesticides_tonnes":pesticides_range,
                            "avg_temp":[temperature]*20})
    pred_pest = np.clip(pipeline.predict(df_pest),0,None)

    # Plotly interactive graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rainfall_range, y=pred_rain, mode='lines+markers', name='Rainfall', line=dict(color='#e63946', width=3)))
    fig.add_trace(go.Scatter(x=temp_range, y=pred_temp, mode='lines+markers', name='Temperature', line=dict(color='#2a9d8f', width=3)))
    fig.add_trace(go.Scatter(x=pesticides_range, y=pred_pest, mode='lines+markers', name='Pesticides', line=dict(color='#f4a261', width=3)))

    fig.update_layout(title=f"Yield Sensitivity for {item}",
                      xaxis_title="Input Value", yaxis_title="Predicted Yield (tonnes)",
                      template="plotly_white", font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<div style='text-align:center; color:#17652a; padding:10px; font-size:14px'>
    Made with ❤️ by Siddhant | Powered by Python & Streamlit
</div>
""", unsafe_allow_html=True)
