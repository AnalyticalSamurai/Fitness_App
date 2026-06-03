import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Fitness Analytics Engine", layout="wide")
st.title("🏋️‍♂️ Intelligent Fitness & Nutrition Analytics Engine")
st.write("A production-ready data application featuring automated data cleaning, anomaly detection, and predictive insights.")

# --- STEP 1: INITIALIZE MOCK DATA (Simulating a database) ---
if 'fitness_data' not in st.session_state:
    st.session_state.fitness_data = pd.DataFrame([
        {"Date": "2026-05-28", "Weight_Lifted": 100, "Calories": 2500, "Sleep_Hours": 8.0},
        {"Date": "2026-05-29", "Weight_Lifted": 105, "Calories": 2600, "Sleep_Hours": 7.5},
        {"Date": "2026-05-30", "Weight_Lifted": 110, "Calories": 2400, "Sleep_Hours": 5.0}, 
        {"Date": "2026-05-31", "Weight_Lifted": None, "Calories": 2550, "Sleep_Hours": 8.0}, 
        {"Date": "2026-06-01", "Weight_Lifted": 115, "Calories": 2700, "Sleep_Hours": 7.8},
    ])

# --- STEP 2: USER INPUT INTERFACE ---
st.subheader("📥 Log Today's Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    input_date = st.date_input("Date", datetime.date.today())
with col2:
    input_weight = st.text_input("Max Bench Press Weight (e.g., 120 lbs)", value="120")
with col3:
    input_calories = st.number_input("Calories Consumed", min_value=1000, max_value=6000, value=2500)
with col4:
    input_sleep = st.number_input("Sleep Quality (Hours)", min_value=0.0, max_value=24.0, value=7.0)

if st.button("Submit Metrics to Data Pipeline"):
    new_row = {
        "Date": str(input_date),
        "Weight_Lifted": input_weight,
        "Calories": input_calories,
        "Sleep_Hours": input_sleep
    }
    st.session_state.fitness_data = pd.concat([st.session_state.fitness_data, pd.DataFrame([new_row])], ignore_index=True)
    st.success("Data successfully injected into the pipeline!")

# --- STEP 3: THE DATA CLEANING & INTEGRITY PIPELINE ---
# We create a clean, isolated dataframe copy to process
df_clean = st.session_state.fitness_data.copy()

# 1. Clean Text Inputs: Strip out characters like "lbs", "kg", spaces, and force to numeric safely
df_clean['Weight_Lifted'] = df_clean['Weight_Lifted'].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True)
df_clean['Weight_Lifted'] = pd.to_numeric(df_clean['Weight_Lifted'], errors='coerce')

# 2. Impute Missing Values: Forward fill, then backward fill if the first entry is missing
df_clean['Weight_Lifted'] = df_clean['Weight_Lifted'].ffill().bfill()

# 3. CRITICAL FIX: Convert text dates to true DateTime objects so sorting works seamlessly over months/years
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean = df_clean.sort_values("Date").reset_index(drop=True)

# --- STEP 4: ANALYTICS & ANOMALY DETECTION ENGINE ---
# Calculate a rolling 3-entry average for performance trajectory
df_clean['Weight_Rolling_Avg'] = df_clean['Weight_Lifted'].rolling(window=3, min_periods=1).mean()

# Detect Burnout Risk (Anomaly: Current training volume is at/above average, but sleep is critically low)
latest_entry = df_clean.iloc[-1]
is_burnout_risk = False
if (latest_entry['Sleep_Hours'] < 6.0) and (latest_entry['Weight_Lifted'] >= df_clean['Weight_Lifted'].mean()):
    is_burnout_risk = True

# --- STEP 5: VISUAL EXECUTIVE DASHBOARD ---
st.markdown("---")
st.subheader("📊 Live Performance Dashboard")

# Metric Cards Display
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Current Strength Peak", f"{int(latest_entry['Weight_Lifted'])} lbs")
m_col2.metric("Target Calories", f"{int(latest_entry['Calories'])} kcal")
m_col3.metric("Last Sleep Session", f"{latest_entry['Sleep_Hours']} Hours")

# AI Insights & Anomaly Warning Box
st.write("### 🧠 Automated System Insights")
# Format date nicely for the UI message
display_date = latest_entry['Date'].strftime('%Y-%m-%d')

if is_burnout_risk:
    st.error(f"⚠️ **Anomaly Warning (Burnout Risk Detected):** On {display_date}, your training volume was high ({int(latest_entry['Weight_Lifted'])} lbs) but your sleep fell to {latest_entry['Sleep_Hours']} hours. The model flags this configuration as a high statistical risk for overtraining or plateauing.")
else:
    st.success("✅ **System Status Nominal:** Your training volume aligns beautifully with your biological recovery patterns. Keep it up!")

# Plotly Interactive Chart
fig = px.line(df_clean, x="Date", y=["Weight_Lifted", "Weight_Rolling_Avg"], 
              labels={"value": "Weight (lbs)", "variable": "Metric", "Date": "Timeline"},
              title="Strength Progression Over Time (Raw Data vs. Rolling Operational Trend)",
              markers=True)
fig.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# Show clean underlying data table
with st.expander("🔍 View Cleaned Production Dataset (Pandas Output)"):
    # Format the date column visually just for the display dataframe table
    df_display = df_clean.copy()
    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(df_display)
    