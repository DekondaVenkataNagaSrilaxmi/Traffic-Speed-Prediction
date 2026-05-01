import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load predictions
df = pd.read_csv("predictions.csv")
sensor_count = (len(df.columns) - 1) // 2

low, medium, high = 0, 0, 0

for i in range(sensor_count):
    actual = df[f'sensor_{i}_actual']
    pred = df[f'sensor_{i}_pred']
    mae = mean_absolute_error(actual, pred)

    if mae < 0.5:
        low += 1
    elif mae < 1.0:
        medium += 1
    else:
        high += 1

# Pie chart
labels = ['Low Error (<0.5)', 'Medium Error (0.5–1.0)', 'High Error (≥1.0)']
sizes = [low, medium, high]
colors = ['#2ecc71', '#f1c40f', '#e74c3c']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Distribution of Sensors by MAE Range")
plt.axis('equal')
plt.tight_layout()
plt.show()
