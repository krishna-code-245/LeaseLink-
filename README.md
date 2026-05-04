# Install dependencies before running:
# pip install flask kagglehub scikit-learn pandas matplotlib

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template_string, request
import numpy as np

# -------------------------------
# Step 1: Load and Train AI Model
# -------------------------------
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "kunwarakash/chennai-housing-sales-price"
)

# Features and target
X = df[['Area', 'BHK']]
y = df['Sale Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# Step 2: Flask Web App
# -------------------------------
app = Flask(__name__)

# HTML template (embedded directly)
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Rental AI Predictor</title>
    <style>
        body { font-family: Arial; margin: 40px; background-color: #f9f9f9; }
        h2 { color: #333; }
        form { margin-bottom: 20px; }
        button { background-color: #4CAF50; color: white; padding: 8px 12px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
    </style>
</head>
<body>
    <h2>Chennai Housing Price Prediction</h2>
    <form action="/predict" method="post">
        <label>Area (sqft):</label>
        <input type="text" name="area" required><br><br>
        <label>BHK:</label>
        <input type="text" name="bhk" required><br><br>
        <button type="submit">Predict</button>
    </form>
    <h3>{{ prediction_text }}</h3>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(html_template)

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bhk = int(request.form["bhk"])
    features = np.array([[area, bhk]])
    prediction = model.predict(features)[0]
    return render_template_string(html_template, prediction_text=f"Predicted Price: ₹{prediction:,.2f}")

# -------------------------------
# Step 3: Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
