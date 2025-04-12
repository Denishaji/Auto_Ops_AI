import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess(df):
    df = df.copy()
    df.drop(columns=["customerID"], inplace=True)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

def train_and_save_model(data_path, model_path):
    df = pd.read_csv(data_path)
    df = preprocess(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved to {model_path}")

# Run the training
if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    train_and_save_model("data/historical.csv", "model/model_v1.pkl")

