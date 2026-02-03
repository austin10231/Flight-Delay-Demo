import math
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

# =========================
# 1) Put AIRPORT_COORDS HERE
# =========================
# 把你那一大段 AIRPORT_COORDS = {...} 粘贴到这里
# =========================
# US Airport Coordinates
# =========================
AIRPORT_COORDS = {
    # Major hubs
    "ATL": (33.6407, -84.4277),
    "LAX": (33.9416, -118.4085),
    "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737),
    "JFK": (40.6413, -73.7781),
    "SFO": (37.6213, -122.3790),
    "SEA": (47.4502, -122.3088),
    "LAS": (36.0840, -115.1537),
    "MCO": (28.4312, -81.3081),
    "CLT": (35.2140, -80.9431),
    "EWR": (40.6895, -74.1745),
    "PHX": (33.4342, -112.0116),
    "IAH": (29.9902, -95.3368),
    "MIA": (25.7959, -80.2870),
    "BOS": (42.3656, -71.0096),
    "MSP": (44.8848, -93.2223),
    "DTW": (42.2162, -83.3554),
    "PHL": (39.8744, -75.2424),
    "LGA": (40.7769, -73.8740),
    "BWI": (39.1754, -76.6684),
    "DCA": (38.8512, -77.0402),

    # West
    "SAN": (32.7338, -117.1933),
    "SJC": (37.3639, -121.9289),
    "OAK": (37.7126, -122.2197),
    "SMF": (38.6951, -121.5908),
    "PDX": (45.5898, -122.5951),
    "SNA": (33.6757, -117.8682),
    "BUR": (34.2007, -118.3587),
    "ONT": (34.0559, -117.6005),
    "PSP": (33.8297, -116.5067),

    # Central
    "AUS": (30.1975, -97.6664),
    "DAL": (32.8471, -96.8517),
    "HOU": (29.6454, -95.2789),
    "SAT": (29.5337, -98.4698),
    "STL": (38.7500, -90.3700),
    "MCI": (39.2976, -94.7139),
    "OMA": (41.3030, -95.8941),

    # East / South
    "TPA": (27.9755, -82.5332),
    "FLL": (26.0742, -80.1506),
    "RSW": (26.5362, -81.7552),
    "JAX": (30.4941, -81.6879),
    "RDU": (35.8801, -78.7880),
    "GSO": (36.0978, -79.9373),
    "BNA": (36.1245, -86.6782),
    "MEM": (35.0424, -89.9767),

    # Midwest / Others
    "CLE": (41.4117, -81.8498),
    "CMH": (39.9980, -82.8919),
    "CVG": (39.0488, -84.6678),
    "PIT": (40.4915, -80.2329),
    "IND": (39.7173, -86.2944),
    "GRR": (42.8808, -85.5228),

    # Mountain
    "SLC": (40.7899, -111.9791),
    "BOI": (43.5644, -116.2228),
    "ABQ": (35.0402, -106.6090),
    "ELP": (31.8072, -106.3780),
}


MODEL_PATH = "model/flight_delay_catboost.cbm"

@st.cache_resource
def load_model():
    m = CatBoostRegressor(verbose=False)
    m.load_model(MODEL_PATH)
    return m

def haversine_distance(origin: str, dest: str) -> float:
    origin = origin.upper()
    dest = dest.upper()
    if origin not in AIRPORT_COORDS or dest not in AIRPORT_COORDS:
        raise ValueError(f"Unknown airport code(s): {origin}, {dest}")

    lat1, lon1 = AIRPORT_COORDS[origin]
    lat2, lon2 = AIRPORT_COORDS[dest]

    R = 3958.8  # miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def estimate_arrival_time(dep_hour: int, dep_min: int, distance_miles: float):
    # cruise speed ~500 mph + 30 min buffer
    total_minutes = int(round(distance_miles / 500.0 * 60 + 30))
    dep_total = int(dep_hour) * 60 + int(dep_min)
    arr_total = (dep_total + total_minutes) % (24 * 60)
    return arr_total // 60, arr_total % 60

FEATURE_COLUMNS = [
    "year","month","day_of_week",
    "op_unique_carrier","origin","dest",
    "distance","cancelled","diverted",
    "dep_hour","dep_min","arr_hour","arr_min"
]

def predict_delay(carrier, origin, dest, dep_hour, dep_min=0, year=2024, month=1, day_of_week=1):
    model = load_model()
    distance = haversine_distance(origin, dest)
    arr_hour, arr_min = estimate_arrival_time(dep_hour, dep_min, distance)

    input_data = {
        "year": int(year),
        "month": int(month),
        "day_of_week": int(day_of_week),
        "op_unique_carrier": carrier.upper(),
        "origin": origin.upper(),
        "dest": dest.upper(),
        "distance": float(distance),
        "cancelled": 0,
        "diverted": 0,
        "dep_hour": int(dep_hour),
        "dep_min": int(dep_min),
        "arr_hour": int(arr_hour),
        "arr_min": int(arr_min),
    }

    X = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    pred = float(model.predict(X)[0])

    return pred, distance, f"{arr_hour:02d}:{arr_min:02d}"

# =========================
# Streamlit UI (put at bottom)
# =========================
st.set_page_config(page_title="Flight Delay Prediction Demo", page_icon="✈️")
st.title("✈️ Flight Delay Prediction Demo")

carrier = st.text_input("Carrier (e.g., AA)", "AA").upper()
origin = st.text_input("Origin Airport (e.g., SFO)", "SFO").upper()
dest = st.text_input("Destination Airport (e.g., LAX)", "LAX").upper()
dep_hour = st.slider("Departure Hour", 0, 23, 10)
dep_min = st.slider("Departure Minute", 0, 59, 0)

if st.button("Predict"):
    try:
        pred, dist, arr_time = predict_delay(
            carrier=carrier, origin=origin, dest=dest,
            dep_hour=dep_hour, dep_min=dep_min,
            year=2024, month=1, day_of_week=1
        )
        st.caption(f"Estimated distance: {dist:.0f} miles")
        st.caption(f"Derived arrival time: {arr_time}")
        st.success(f"Predicted Delay: {pred:.1f} minutes")
        if pred < 0:
            st.info("Negative means expected early arrival.")
    except Exception as e:
        st.error(str(e))
