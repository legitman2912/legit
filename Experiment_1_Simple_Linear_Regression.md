# Experiment 1: Simple Linear Regression (California Housing Dataset)

## Aim
To implement Simple Linear Regression using one feature (`total_rooms`) from the California Housing dataset (`housing.csv`) to predict house prices using the top 150 records.

## Python Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("/content/sample_data/housing.csv")

# -----------------------------
# Select feature and target
# -----------------------------
X = df['total_rooms'].head(150).values.reshape(-1, 1)
y = df['median_house_value'].head(150).values.reshape(-1, 1)

# -----------------------------
# Visualization (Scatter Plot)
# -----------------------------
plt.scatter(X, y, color="blue", s=20)

# -----------------------------
# Mean values
# -----------------------------
m_x = np.mean(X)
m_y = np.mean(y)

# -----------------------------
# Compute coefficients
# -----------------------------
SS_xy = np.sum((X - m_x) * (y - m_y))
SS_xx = np.sum((X - m_x) * (X - m_x))

b1 = SS_xy / SS_xx
b0 = m_y - b1 * m_x

print("Slope (b1):", b1)
print("Intercept (b0):", b0)

# -----------------------------
# Predictions
# -----------------------------
y_pred = b0 + b1 * X

# -----------------------------
# Manual MSE
# -----------------------------
n = len(y)
mse = np.sum((y - y_pred) ** 2) / n
print("MSE:", mse)

# -----------------------------
# Manual R^2
# -----------------------------
SS_res = np.sum((y - y_pred) ** 2)
SS_tot = np.sum((y - m_y) ** 2)
r2 = 1 - (SS_res / SS_tot)
print("R^2:", r2)

# -----------------------------
# Plot regression line
# -----------------------------
plt.plot(X, y_pred, color="red")
plt.xlabel("Total Rooms")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression (150 Samples)")
plt.show()
```

## Output

```
Slope (b1): 30.03063864613546
Intercept (b0): 144527.3970744946
MSE: 9070344426.642298
R²: 0.1494586436630212
```
