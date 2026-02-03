import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

st.title("✈️ Flight Delay Prediction Demo")

# Load model
model = CatBoostRegressor()
model.load_model("model/flight_delay_catboost.cbm")

# User inputs (只让用户填 3 个)
carrier = st.text_input("Carrier", "AA")
origin = st.text_input("Origin", "SFO")
dest = st.text_input("Dest", "LAX")

if st.button("Predict"):
    # ---- Build FULL feature vector (must match training) ----
    input_data = {
        "year": 2024,
        "month": 1,
        "day_of_week": 1,

        "op_unique_carrier": carrier,
        "origin": origin,
        "dest": dest,

        "distance": 337.0,    # 示例值
        "cancelled": 0,
        "diverted": 0,

        "dep_hour": 10,
        "dep_min": 0,
        "arr_hour": 12,
        "arr_min": 0
    }

    FEATURE_COLUMNS = [
        "year",
        "month",
        "day_of_week",
        "op_unique_carrier",
        "origin",
        "dest",
        "distance",
        "cancelled",
        "diverted",
        "dep_hour",
        "dep_min",
        "arr_hour",
        "arr_min"
    ]

    X = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    y = model.predict(X)[0]
    st.success(f"Predicted Delay: {y:.1f} minutes")
