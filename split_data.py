import pandas as pd

df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Clean missing TotalCharges
df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Simulate drift via contract type
historical_df = df[df['Contract'] == 'Month-to-month']
latest_df = df[df['Contract'].isin(['One year', 'Two year'])]

# Save them
historical_df.to_csv("data/historical.csv", index=False)
latest_df.to_csv("data/latest.csv", index=False)

print("Data prepared and saved.")

