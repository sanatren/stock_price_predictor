import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler

# ✅ **Set Page Configuration**
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# ✅ **Paths**
DATA_PATH = "../data/processed/final_merged_stock_data.csv"
FEATURES_DIR = "../data/feature_engineering data/"
MODEL_DIR = "../models/arima_sarima_models/"
BOOSTING_MODEL_DIR = "../models/boosting_models/"
MODEL_COMPARISON_FILE = "../data/final_model_comparison.csv"





# ✅ **Load Processed Data**
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])

# ✅ **Sidebar Navigation**
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis (EDA)", "Stock Trends", "Predict Prices", "Model Comparison", "Conclusion"])

# ✅ **Load Data**
df = load_data()
stocks = [col.split("_")[0] for col in df.columns if "_Close" in col]

# Load ML Models
@st.cache_resource
def load_ml_model(stock, model_type):
    model_path = os.path.join(BOOSTING_MODEL_DIR, f"{stock}_{model_type}.pkl")
    return joblib.load(model_path) if os.path.exists(model_path) else None

# ✅ Load Model Comparison Results
@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_FILE) if os.path.exists(MODEL_COMPARISON_FILE) else None

# ✅ Compute Best Models
@st.cache_data
def compute_best_models():
    model_df = load_model_comparison()
    if model_df is None:
        return {}

    best_models = model_df.loc[model_df.groupby("stock")["rmse"].idxmin(), ["stock", "model"]]
    return dict(zip(best_models["stock"], best_models["model"]))

# ✅ Load Best Models Before Predictions
best_models = compute_best_models()


# Load ARIMA & SARIMA Models
@st.cache_resource
def load_arima_sarima_models(stock):
    arima_path = os.path.join(MODEL_DIR, f"{stock}_ARIMA.pkl")
    sarima_path = os.path.join(MODEL_DIR, f"{stock}_SARIMA.pkl")

    arima_model = joblib.load(arima_path) if os.path.exists(arima_path) else None
    sarima_model = joblib.load(sarima_path) if os.path.exists(sarima_path) else None

    return arima_model, sarima_model

def recursive_ml_forecast(model, last_row, days, stock):
    forecast = []
    current_input = last_row.copy()

    model_features = model.feature_names_in_
    current_input = current_input.reindex(model_features, fill_value=0)  # ✅ Fix: Removed `columns=`

    for _ in range(days):
        predicted_price = model.predict([current_input.values])[0]
        forecast.append(predicted_price)

        # ✅ Update lag features dynamically
        if f"{stock}_Close_Lag_1" in current_input.index:
            current_input[f"{stock}_Close_Lag_1"] = predicted_price
    
    return forecast


# ✅ **Home Page**
if page == "Home":
    st.title("📈 Stock Price Prediction App")
    st.write("""
    - 📊 Visualize historical stock trends  
    - 🔮 Predict future stock prices  
    - 🏆 Compare different model performances  
    - 📊 Perform detailed **EDA (Exploratory Data Analysis)**
    """)

    st.subheader("📌 Available Stocks")
    st.write(", ".join(stocks))

elif page == "Exploratory Data Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis (EDA)")

    stock = st.selectbox("Select a stock:", stocks)
    col_name = f"{stock}_Close"

    st.subheader(f"📉 {stock} Stock Price Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df[col_name], label=f"{stock} Closing Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{stock} Stock Price Over Time")
    ax.legend()
    st.pyplot(fig)

    # ✅ **Rolling Mean & Standard Deviation**
    st.subheader("📊 Rolling Mean & Standard Deviation")
    rolling_window = st.slider("Select Rolling Window:", 5, 60, 30)

    df[f"{stock}_RollingMean"] = df[col_name].rolling(rolling_window).mean()
    df[f"{stock}_RollingStd"] = df[col_name].rolling(rolling_window).std()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df[col_name], label="Actual Price", color="blue")
    ax.plot(df["Date"], df[f"{stock}_RollingMean"], label="Rolling Mean", linestyle="dashed", color="orange")
    ax.plot(df["Date"], df[f"{stock}_RollingStd"], label="Rolling Std", linestyle="dashed", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{stock} - Rolling Mean & Std (Window={rolling_window})")
    ax.legend()
    st.pyplot(fig)

    # ✅ **Histogram & Distribution**
    st.subheader("📊 Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[col_name].dropna(), bins=50, kde=True, color="blue", ax=ax)
    ax.set_title(f"{stock} Price Distribution")
    st.pyplot(fig)

    # ✅ **Correlation Heatmap**
    st.subheader("📊 Correlation Heatmap")
    corr_matrix = df[[col for col in df.columns if stock in col]].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ✅ **Stationarity Test (ADF Test)**
    st.subheader("📉 ADF Test (Stationarity Check)")
    adf_result = adfuller(df[col_name].dropna())
    st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
    st.write(f"**p-value:** {adf_result[1]:.4f}")
    st.write("✅ **Stationary**" if adf_result[1] < 0.05 else "❌ **Non-Stationary**")

# ✅ **Stock Trends Page**
elif page == "Stock Trends":
    st.title("📉 Stock Price Trends")
    stock = st.selectbox("Select a stock:", stocks)
    col_name = f"{stock}_Close"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df[col_name], label=f"{stock} Closing Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{stock} Stock Price Over Time")
    ax.legend()
    st.pyplot(fig)

# ✅ **Stock Trends Page**
elif page == "Stock Trends":
    st.title("📉 Stock Price Trends")
    stock = st.selectbox("Select a stock:", stocks)
    col_name = f"{stock}_Close"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df[col_name], label=f"{stock} Closing Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{stock} Stock Price Over Time")
    ax.legend()
    st.pyplot(fig)

# ✅ **Predict Prices Page**
elif page == "Predict Prices":
    st.title("🔮 Predict Future Stock Prices")
    stock = st.selectbox("Select a stock for prediction:", stocks)
    days = st.slider("Select prediction days:", 1, 30, 7)

    best_model = best_models.get(stock, "Gradient Boosting")

    feature_file = os.path.join(FEATURES_DIR, f"{stock}_boosting_features.csv")
    feature_df = pd.read_csv(feature_file)
    last_row = feature_df.iloc[-1].drop("Date")

    # Load models
    arima_model, sarima_model = load_arima_sarima_models(stock)
    best_ml_model = load_ml_model(stock, best_model)

    # Make Predictions
    predictions = {"Date": pd.date_range(df["Date"].max(), periods=days + 1, freq='D')[1:]}

    if arima_model:
        predictions["ARIMA"] = arima_model.forecast(steps=days)
    if sarima_model:
        predictions["SARIMA"] = sarima_model.forecast(steps=days)

    if best_ml_model:
        predictions[best_model] = recursive_ml_forecast(best_ml_model, last_row, days, stock)

    pred_df = pd.DataFrame(predictions)


    # ✅ Display Predictions
    st.subheader("📊 Forecasted Prices")
    st.dataframe(pred_df)

     # ✅ Explanation Message
    st.markdown("""
    ### 📢 Why is ARIMA/SARIMA performing better?
    - ARIMA/SARIMA models continuously **adapt to new trends and correct past errors**, making them **more reliable** for time series forecasting.
    - ML models (LightGBM, XGBoost, Gradient Boosting) perform well in training but **fail in real-world forecasting** because they assume **stock prices follow a fixed pattern**.
    - Machine Learning models struggle with **dynamically updating lag features**, leading to **flat predictions**.
    """)

    # ✅ Plot Predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df[f"{stock}_Close"], label="Historical", color="blue")
    for model in predictions.keys():
        if model != "Date":
            ax.plot(pred_df["Date"], pred_df[model], label=model, linestyle="dashed")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{stock} Price Prediction ({best_model})")
    ax.legend()
    st.pyplot(fig)


# ✅ **Model Comparison Page**
elif page == "Model Comparison":
    st.title("🏆 Model Performance Comparison")

    eval_df = pd.read_csv(MODEL_COMPARISON_FILE) if os.path.exists(MODEL_COMPARISON_FILE) else None

    if eval_df is not None:
        st.dataframe(eval_df)

        # 📊 **RMSE Comparison**
        st.subheader("📉 RMSE Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        eval_df.groupby("model")["rmse"].mean().plot(kind="bar", ax=ax, color="blue")
        ax.set_ylabel("RMSE")
        ax.set_title("📊 Model RMSE Comparison (Lower is better)")
        st.pyplot(fig)

        # 📊 **R² Score Comparison**
        st.subheader("📈 R² Score Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        eval_df.groupby("model")["r2"].mean().plot(kind="bar", ax=ax, color="green")
        ax.set_ylabel("R² Score")
        ax.set_title("📊 Model R² Comparison (Higher is better)")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Model comparison data not found!")

# ✅ **Conclusion Page**
elif page == "Conclusion":
    st.title("🚀 Conclusion & Next Steps")
    st.write("""
    - **EDA (Exploratory Data Analysis)** provides insights into trends, seasonality, and correlation.
    - **Machine Learning models like XGBoost & LightGBM** perform well in training but often struggle with real-world forecasting.
    - **ARIMA/SARIMA models tend to generalize better for time-series data** because they adapt dynamically to changing trends.
    """)

   



