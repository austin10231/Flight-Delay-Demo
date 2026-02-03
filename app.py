import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

st.title("✈️ Flight Delay Prediction Demo")

model = CatBoostRegressor()
model.load_model("model/flight_delay_catboost.cbm")  # 注意：这里本地要有这个文件才能跑

carrier = st.text_input("Carrier", "AA")
origin = st.text_input("Origin", "SFO")
dest = st.text_input("Dest", "LAX")

if st.button("Predict"):
    X = pd.DataFrame([{
        "op_unique_carrier": carrier,
        "origin": origin,
        "dest": dest
    }])
    y = model.predict(X)[0]
    st.success(f"Predicted delay: {y:.1f} minutes")
