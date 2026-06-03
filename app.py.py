import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Fitness Analytics Engine", layout="wide")
st.title("🏋️‍♂️ Intelligent Fitness & Nutrition Analytics Engine")
st.write("A production-ready data application featuring automated data cleaning, anomaly detection, and predictive insights.")

# --- STEP 1: INITIALIZE MOCK DATA (Simulating a database) ---
# If it's the user's first time, we create a basic dataset with a few missing pieces to show off data cleaning.
if 'fitness_data' not in st.session_state:
    st.session_state.fitness_data = pd.DataFrame([
        {"Date": "2026-05-28", "Weight_Lifted": 100, "Calories": 2500, "Sleep_Hours": 8.0},
        {"Date": "2026-05-29", "Weight_Lifted": 105, "Calories": 2600, "Sleep_Hours": 7.5},
        {"Date": "2026-05-30", "Weight_Lifted": 110, "Calories": 2400, "Sleep_Hours": 5.0}, # High volume, low sleep day
        {"Date": "2026-05-31", "Weight_Lifted": None, "Calories": 2550, "Sleep_Hours": 8.0}, # Missing entry simulation
        {"Date": "2026-06-01", "Weight_Lifted": 115, "Calories": 2700, "Sleep_Hours": 7.8},
    ])

# --- STEP 2: USER INPUT INTERFACE ---
st.subheader("📥 Log Today's Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    input_date = st.date_input("Date", datetime.date.today())
with col2:
    # We accept text to simulate cleaning up messy inputs (e.g., stripping "lbs")
    input_weight = st.text_input("Max Bench Press Weight (e.g., 120 lbs)", value="120")
with col3:
    input_calories = st.number_input("Calories Consumed", min_value=1000, max_value=6000, value=2500)
with col4:
    input_sleep = st.number_input("Sleep Quality (Hours)", min_value=0.0, max_value=24.0, value=7.0)

if st.button("Submit Metrics to Data Pipeline"):
    # Create a new row from user inputs
    new_row = {
        "Date": str(input_date),
        "Weight_Lifted": input_weight,
        "Calories": input_calories,
        "Sleep_Hours": input_sleep
    }
    # Append to our dataset
    st.session_state.fitness_data = pd.concat([st.session_state.fitness_data, pd.DataFrame([new_row])], ignore_index=True)
    st.success("Data sent to backend pipeline!")

# --- STEP 3: THE DATA CLEANING & INTEGRITY PIPELINE ---
df_raw = st.session_state.fitness_data.copy()

# 1. Clean Text Inputs: Remove "lbs", "kg", spaces, and force to numeric
if df_raw['Weight_Lifted'].dtype == 'O': # If it's stored as text/object
    df_raw['Weight_Lifted'] = df_raw['Weight_Lifted'].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True)
df_raw['Weight_Lifted'] = pd.to_numeric(df_raw['Weight_Lifted'])

# 2. Impute Missing Values: If a user misses a lifting log, fill it with the previous day's weight
df_raw['Weight_Lifted'] = df_raw['Weight_Lifted'].ffill().bfill()

# Ensure sorted dates for accurate rolling logic
df_raw = df_raw.sort_values("Date").reset_index(drop=True)

# --- STEP 4: ANALYTICS & ANOMALY DETECTION ENGINE ---
# Calculate a 3-day rolling average for performance trajectory
df_raw['Weight_Rolling_Avg'] = df_raw['Weight_Lifted'].rolling(window=3, min_periods=1).mean()

# Detect Burnout Risk (Anomaly: High lifting volume + Low sleep)
latest_entry = df_raw.iloc[-1]
is_burnout_risk = False
if (latest_entry['Sleep_Hours'] < 6.0) and (latest_entry['Weight_Lifted'] >= df_raw['Weight_Lifted'].mean()):
    is_burnout_risk = True

# --- STEP 5: VISUAL EXECUTIVE DASHBOARD ---
st.markdown("---")
st.subheader("📊 Live Performance Dashboard")

# Metric Cards Display
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Current Strength Peak", f"{latest_entry['Weight_Lifted']} lbs")
m_col2.metric("Target Calories", f"{latest_entry['Calories']} kcal")
m_col3.metric("Last Sleep Session", f"{latest_entry['Sleep_Hours']} Hours")

# AI Insights & Anomaly Warning Box
st.write("### 🧠 Automated System Insights")
if is_burnout_risk:
    st.error(f"⚠️ **Anomaly Warning (Burnout Risk Detected):** On {latest_entry['Date']}, your training volume was high ({latest_entry['Weight_Lifted']} lbs) but your sleep fell to {latest_entry['Sleep_Hours']} hours. The model flags this as a high risk for injury or plateauing.")
else:
    st.success("✅ **System Status Nominal:** Your training volume aligns well with your recovery patterns. Keep it up!")

# Plotly Interactive Chart
fig = px.line(df_raw, x="Date", y=["Weight_Lifted", "Weight_Rolling_Avg"], 
              labels={"value": "Weight (lbs)", "variable": "Metric"},
              title="Strength Progression Over Time (Raw vs. Rolling Trend)",
              markers=True)
fig.update_layout(template="plotly_dark" if False else "plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Show clean underlying data table
with st.expander("🔍 View Cleaned Production Dataset"):
    st.dataframe(df_raw)
    