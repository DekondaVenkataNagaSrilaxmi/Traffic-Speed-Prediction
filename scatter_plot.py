import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the predictions
df = pd.read_csv("predictions.csv")

# Load normalization stats
norm_stats = np.load("data/METR-LA/norm_stats.npz")
mean = norm_stats["mean"]     # shape: (207,)
std = norm_stats["std"]       # shape: (207,)

# Choose a sensor ID (e.g. best-performing)
sensor_id = 65

# Denormalize actual and predicted
actual_norm = df[f"sensor_{sensor_id}_actual"]
pred_norm = df[f"sensor_{sensor_id}_pred"]

actual_kmh = actual_norm * std[sensor_id] + mean[sensor_id]
pred_kmh = pred_norm * std[sensor_id] + mean[sensor_id]

# Create scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(actual_kmh, pred_kmh, alpha=0.5, color='green', edgecolors='k')

# Line of perfect prediction
min_val = min(actual_kmh.min(), pred_kmh.min())
max_val = max(actual_kmh.max(), pred_kmh.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

plt.title(f"Actual vs Predicted Speed (Sensor {sensor_id}) [km/h]")
plt.xlabel("Actual Speed (km/h)")
plt.ylabel("Predicted Speed (km/h)")
plt.grid(True)
plt.tight_layout()
plt.show()
