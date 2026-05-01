import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("predictions.csv")
sensor_count = (len(df.columns) - 1) // 2

mae_list = []
for i in range(sensor_count):
    actual = df[f'sensor_{i}_actual']
    pred = df[f'sensor_{i}_pred']
    mae = mean_absolute_error(actual, pred)
    mae_list.append(mae)

plt.figure(figsize=(12, 5))
plt.hist(mae_list, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of MAE for All Sensors")
plt.xlabel("Mean Absolute Error (MAE)")
plt.ylabel("Number of Sensors")
plt.tight_layout()
plt.show()
