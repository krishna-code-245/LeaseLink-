#lease-link-ai
# Step 1: Install required libraries
# pip install kagglehub scikit-learn pandas matplotlib

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 2: Load dataset from Kaggle
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "kunwarakash/chennai-housing-sales-price"
)

print("First 5 records:", df.head())

# Step 3: Select features (inputs) and target (output)
# Example: predicting 'Sale Price' based on area, bedrooms, and location
X = df[['Area', 'BHK', 'Location']]  # features
y = df['Sale Price']                 # target

# Convert categorical data (like Location) into numbers
X = pd.get_dummies(X, drop_first=True)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train a machine learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
predictions = model.predict(X_test)

# Step 7: Show results
print("Predicted Prices:", predictions[:10])
print("Actual Prices:", list(y_test[:10]))

# Step 8: Visualize predictions vs actual
plt.scatter(y_test, predictions)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Chennai Housing Price Prediction")
plt.show()
