import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from training.train_model import train_and_save_model
from drift_detection.drift_checker import detect_drift

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

def agent_step():
    # Check for drift
    drift = detect_drift("data/historical.csv", "data/latest.csv")
    if not drift:
        print("✅ No drift detected. No action needed.")
        return

    print("⚠️ Drift detected. Retraining new model...")

    # Load datasets
    df_old = pd.read_csv("data/historical.csv")
    df_new = pd.read_csv("data/latest.csv")

    def preprocess(df):
        df = df.copy()
        df.drop(columns=["customerID"], inplace=True)
        from sklearn.preprocessing import LabelEncoder
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = LabelEncoder().fit_transform(df[col])
        return df

    df_old = preprocess(df_old)
    df_new = preprocess(df_new)

    X_new = df_new.drop("Churn", axis=1)
    y_new = df_new["Churn"]

    # Load old model
    old_model = joblib.load("model/model_v1.pkl")
    old_acc = evaluate_model(old_model, X_new, y_new)

    # Train new model on latest data
    os.makedirs("model", exist_ok=True)
    train_and_save_model("data/latest.csv", "model/model_v2.pkl")
    new_model = joblib.load("model/model_v2.pkl")
    new_acc = evaluate_model(new_model, X_new, y_new)

    # Decision logic
    decision_log = {
        "drift_columns": [col for col, _ in drift],
        "old_accuracy": round(old_acc, 4),
        "new_accuracy": round(new_acc, 4)
    }

    if new_acc > old_acc:
        joblib.dump(new_model, "model/model_v1.pkl")  # Promote
        decision_log["action"] = "✅ New model deployed"
    else:
        decision_log["action"] = "⏸️ Kept existing model"

    # Save log
    with open("logs/agent_log.json", "w") as f:
        json.dump(decision_log, f, indent=2)

    print("🤖 Decision Log:", json.dumps(decision_log, indent=2))

# Run it standalone
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    agent_step()

