import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template_string, request
import numpy as np

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
