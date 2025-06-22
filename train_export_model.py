# train_export_model.py

import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example dataset (use integers or scale to integers)
X = np.array([[1, 5], [4, 9], [8, 3], [2, 7]])
y = np.array([10, 25, 18, 14])

# Train simple linear regression
model = LinearRegression()
model.fit(X, y)

# Predict on training data
y_pred = model.predict(X)

# Compute RMSE (manual way — works for all sklearn versions)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Export weights + bias
weights = model.coef_.tolist()
bias = model.intercept_.item()

# Save to JSON (rounded to integers for circuit compatibility)
model_data = {
    "weights": [int(round(w)) for w in weights],
    "bias": int(round(bias))
}

with open("model.json", "w") as f:
    json.dump(model_data, f)

# Print results
print("✅ Model exported to model.json")
print(f"✅ Training RMSE: {rmse:.4f}")
