import pandas as pd
import json
from datetime import datetime, timezone
import os

# Define base directories
base_dir = "../"  
processed_dir = os.path.join(base_dir, "data", "processed")

# Define file paths 
csv_path = os.path.join(processed_dir, "cleaned_stock_data.csv")
json_path = os.path.join(processed_dir, "cleaned_stock_data.json")

# Define output file paths
merged_csv_path = os.path.join(processed_dir, "final_merged_stock_data.csv")


# ✅ Function to process and rename CSV columns
def process_csv(csv_path):
    df_csv = pd.read_csv(csv_path)

    if "Date" not in df_csv.columns:
        raise ValueError("🚨 The 'Date' column is missing in the CSV file!")

    # ✅ Automatically detect and parse dates
    df_csv["Date"] = pd.to_datetime(df_csv["Date"], errors="coerce", infer_datetime_format=True)

    # 🚨 **Fix: Identify and remove rows where Date could not be parsed**
    invalid_dates = df_csv["Date"].isna().sum()
    if invalid_dates > 0:
        print(f"⚠️ Warning: {invalid_dates} invalid dates found. These rows will be dropped.")
        df_csv.dropna(subset=["Date"], inplace=True)

    # Forward-fill missing stock prices
    df_csv.ffill(inplace=True)

    return df_csv


# ✅ Function to clean and process JSON
def process_json(json_path):
    with open(json_path, "r") as file:
        json_data = json.load(file)

    flattened_data = []
    for key, values in json_data.items():
        if isinstance(key, str) and key.startswith("(") and key.endswith(")"):
            key_parts = key.strip("()").replace("'", "").split(", ")
            if len(key_parts) == 2:
                metric, ticker = key_parts
            else:
                continue  
        else:
            continue  

        # ✅ Automatically detect and fix date format issues in JSON
        for timestamp, value in values.items():
            try:
                date = datetime.utcfromtimestamp(int(timestamp) / 1000).strftime('%Y-%m-%d')
                flattened_data.append({"Date": date, f"{ticker}_{metric}": value})
            except ValueError:
                continue  

    df_json = pd.DataFrame(flattened_data)

    # ✅ Ensure Date is in the correct format
    df_json["Date"] = pd.to_datetime(df_json["Date"], errors="coerce", infer_datetime_format=True)

    # 🚨 Remove invalid dates
    invalid_json_dates = df_json["Date"].isna().sum()
    if invalid_json_dates > 0:
        print(f"⚠️ Warning: {invalid_json_dates} invalid JSON dates found. These rows will be dropped.")
        df_json.dropna(subset=["Date"], inplace=True)

    df_json_grouped = df_json.groupby("Date").first().reset_index()

    return df_json_grouped


# ✅ Process CSV and JSON
df_csv_cleaned = process_csv(csv_path)
df_json_cleaned = process_json(json_path)

# ✅ Convert Date to datetime64[ns] to ensure proper merging
df_csv_cleaned["Date"] = pd.to_datetime(df_csv_cleaned["Date"], errors="coerce")
df_json_cleaned["Date"] = pd.to_datetime(df_json_cleaned["Date"], errors="coerce")

# ✅ Debug: Check Date Ranges
print("CSV Date Range:", df_csv_cleaned["Date"].min(), "to", df_csv_cleaned["Date"].max())
print("JSON Date Range:", df_json_cleaned["Date"].min(), "to", df_json_cleaned["Date"].max())

# ✅ Merge Data on Date (Ensuring Proper Alignment)
merged_df = pd.merge(df_csv_cleaned, df_json_cleaned, on="Date", how="outer")

# ✅ Fill missing values using forward-fill
merged_df.ffill(inplace=True)

# ✅ Save the final dataset
merged_df.to_csv(merged_csv_path, index=False)
print(f"✅ Merged dataset saved at {merged_csv_path}")

# ✅ Preview merged dataset
print("\n🔍 Merged DataFrame Preview:")
print(merged_df.head())
