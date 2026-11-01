import os
from datetime import date

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Crowd Prediction", layout="centered")

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crowd_prediction_model.joblib")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["pipeline"], bundle["meta"]


pipeline, meta = load_model()
H0, H1 = meta["hour_min"], meta["hour_max"]


def predict(dt, hour):
    row = {
        "year": dt.year, "month": dt.month, "day": dt.day, "day_of_week": dt.dayofweek,
        "day_of_year": dt.dayofyear, "hour": hour, "is_weekend": int(dt.dayofweek >= 5),
    }
    if meta["has_company"]:
        row["company"] = meta["company"]
    return float(pipeline.predict(pd.DataFrame([row])[meta["features"]])[0])


st.title("Customer Crowd Prediction System")
st.subheader("AI-Based Business Demonstration (NCP-3, Team 2)")
st.write(
    "Select a date and an hour. The app creates the same time-based features used in training "
    "and the trained Random Forest Regression model returns a **normalized** customer-traffic estimate."
)
st.info(
    "Prototype trained on historical retail foot-traffic data from Würzburg, Germany. "
    "Results are normalized values, not customer counts, and have not been validated for Hyderabad businesses."
)

c1, c2 = st.columns(2)
selected_date = c1.date_input("Select date", value=date(2024, 11, 25))
selected_hour = c2.slider(f"Select hour of the day ({H0}:00 to {H1}:00 in the training data)", H0, H1, 14)

if st.button("Predict Customer Traffic", type="primary"):
    dt = pd.Timestamp(selected_date)
    value = predict(dt, selected_hour)
    st.success("Prediction generated successfully.")
    m1, m2 = st.columns(2)
    m1.metric("Predicted normalized value", f"{value:.4f}")
    m2.metric("Day type", "Weekend" if dt.dayofweek >= 5 else "Weekday")
    st.progress(min(max(value, 0.0), 1.0))
    st.caption("Normalized model output (0 to 1 scale), not an actual number of customers.")

    st.write(f"**Predicted pattern across the day for {selected_date:%d %B %Y}**")
    profile = pd.DataFrame({"Hour": list(range(H0, H1 + 1)),
                            "Predicted normalized value": [predict(dt, h) for h in range(H0, H1 + 1)]}).set_index("Hour")
    st.line_chart(profile)
    st.write(f"Input details: date {selected_date}, hour {selected_hour}:00.")
    st.warning("This is an estimate based on historical patterns. It should not be treated as a verified forecast for a real business.")

with st.expander("About the model"):
    mt = meta["metrics"]
    st.write(
        f"Model: Random Forest Regression (100 trees, max depth 12). Data: {meta['date_min']} to {meta['date_max']}, "
        f"chronological split ({meta['n_train']:,} training / {meta['n_valid']:,} validation records)."
    )
    st.write(f"Validation MAE {mt['mae']:.4f}, RMSE {mt['rmse']:.4f}, R² {mt['r2']:.4f}.")
    st.write("Features: year, month, day, day of week, day of year, hour, weekend flag" + (", company." if meta["has_company"] else "."))

st.divider()
st.caption("NCP-3 Academic Project | ICFAI Business School, Hyderabad | Machine Learning Demonstration")
