import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Residential Energy Analytics", layout="wide")

st.title("🏠 Residential Energy Analytics Platform")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your energy data (.csv)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["DateTime"])
    df.set_index("DateTime", inplace=True)

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Visualization
    st.subheader("📈 Energy Usage Over Time")
    st.line_chart(df["EnergyUsage"])

    st.subheader("📅 Daily Average Energy Usage")
    daily_avg = df["EnergyUsage"].resample("D").mean()
    st.line_chart(daily_avg)

    # Machine Learning Prediction
    st.subheader("🔮 Energy Usage Prediction with ML")

    if 'Temperature' in df.columns and 'Humidity' in df.columns:
        X = df[["Temperature", "Humidity"]]
        y = df["EnergyUsage"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        st.success(f"✅ Model Trained! Mean Absolute Error: {mae:.4f}")

        st.subheader("📊 Actual vs Predicted")
        result_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": predictions[:len(y_test)]
        })
        st.dataframe(result_df.head())

    # Energy-saving tip
    st.subheader("💡 Energy Saving Insight")
    avg_usage = df["EnergyUsage"].mean()
    peak = df[df["EnergyUsage"] > avg_usage * 1.3]

    if not peak.empty:
        st.warning("⚠ You are using high energy during certain hours. Try to shift some load.")
    else:
        st.info("✅ Your energy usage is efficient!")
else:
    st.info("📌 Please upload a CSV file to get started.")
