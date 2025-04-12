import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.copy()
    df.drop(columns=["customerID"], inplace=True)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def detect_drift(historical_path, latest_path, threshold=0.05):
    df1 = preprocess(pd.read_csv(historical_path))
    df2 = preprocess(pd.read_csv(latest_path))

    drifted_cols = []

    for col in df1.columns:
        if col == "Churn": continue  # Skip label
        stat, p_value = ks_2samp(df1[col], df2[col])
        if p_value < threshold:
            drifted_cols.append((col, round(p_value, 4)))

    return drifted_cols

# Run drift check standalone
if __name__ == "__main__":
    drift = detect_drift("data/historical.csv", "data/latest.csv")
    if drift:
        print("⚠️ Drift detected in:")
        for col, p in drift:
            print(f" - {col} (p = {p})")
    else:
        print("✅ No significant drift detected.")

