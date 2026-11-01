import os
from datetime import date

import pandas as pd
import streamlit as st

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(csv_path):
    df = pd.read_csv(csv_path)
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    d = df[df["category"] == "customers"].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["hour"] = pd.to_numeric(d["hour"], errors="coerce")
    d["target"] = pd.to_numeric(d["target"], errors="coerce")
    d = d.dropna(subset=["date", "hour", "target"]).copy()
    d["year"] = d["date"].dt.year; d["month"] = d["date"].dt.month; d["day"] = d["date"].dt.day
    d["day_of_week"] = d["date"].dt.dayofweek; d["day_of_year"] = d["date"].dt.dayofyear
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d = d.sort_values(["date", "hour"]).reset_index(drop=True)
    num = ["year", "month", "day", "day_of_week", "day_of_year", "hour", "is_weekend"]
    cat = ["company"] if "company" in d.columns else []
    d["time_key"] = d["date"].dt.strftime("%Y-%m-%d") + "_" + d["hour"].astype(str)
    keys = d[["time_key", "date", "hour"]].drop_duplicates().sort_values(["date", "hour"])["time_key"].tolist()
    sp = max(1, min(int(len(keys) * 0.8), len(keys) - 1))
    tr = d["time_key"].isin(set(keys[:sp])); va = d["time_key"].isin(set(keys[sp:]))
    X = d[num + cat]; y = d["target"]
    tf = [("numeric", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num)]
    if cat:
        tf.append(("categorical", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    pipe = Pipeline([("preprocessor", ColumnTransformer(tf, remainder="drop")),
                     ("model", RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1))])
    pipe.fit(X[tr], y[tr])
    p = pipe.predict(X[va])
    m = dict(mae=float(mean_absolute_error(y[va], p)), rmse=float(np.sqrt(mean_squared_error(y[va], p))), r2=float(r2_score(y[va], p)))
    meta = dict(features=num + cat, company=(d["company"].mode()[0] if cat else None), has_company=bool(cat),
                hour_min=int(d["hour"].min()), hour_max=int(d["hour"].max()),
                date_min=str(d["date"].min().date()), date_max=str(d["date"].max().date()),
                n_train=int(tr.sum()), n_valid=int(va.sum()), metrics=m)
    return pipe, meta


st.set_page_config(page_title="Customer Crowd Prediction", layout="centered")

import glob

HERE = os.path.dirname(os.path.abspath(__file__))


def find_csv():
    for root in (HERE, os.getcwd()):
        hits = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
        hits = [h for h in hits if "train" in os.path.basename(h).lower()] or hits
        if hits:
            return hits[0]
    return None


CSV_FILE = find_csv()
if CSV_FILE is None:
    st.error("No CSV file found. Upload train.csv to the same GitHub folder as app.py, then reboot the app.")
    st.write("Files visible to the app:", sorted(os.listdir(HERE)))
    st.stop()


@st.cache_resource(show_spinner="Training the Random Forest model on train.csv (first load only)...")
def load_model():
    return train(CSV_FILE)


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
