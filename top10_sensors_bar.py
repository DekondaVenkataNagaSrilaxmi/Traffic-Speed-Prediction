import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load predictions
df = pd.read_csv("predictions.csv")
sensor_count = (len(df.columns) - 1) // 2

mae_scores = []

# Calculate MAE for each sensor
for i in range(sensor_count):
    actual = df[f'sensor_{i}_actual']
    pred = df[f'sensor_{i}_pred']
    mae = mean_absolute_error(actual, pred)
    mae_scores.append((i, mae))

# Sort by MAE (ascending)
mae_scores.sort(key=lambda x: x[1])

# Select top 10
top10 = mae_scores[:10]
sensor_ids = [f"Sensor {i}" for i, _ in top10]
mae_values = [score for _, score in top10]

# Plot
plt.figure(figsize=(10, 5))
plt.bar(sensor_ids, mae_values, color='#3498db')
plt.xlabel("Sensor")
plt.ylabel("MAE")
plt.title("Top 10 Best Performing Sensors (Lowest MAE)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
