import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression

df = pd.read_csv('phoneprice.csv')

# One-hot encoding
df = pd.concat([df, pd.get_dummies(df['Brand'], prefix='Brand', dtype=int )], axis=1)
df.drop(columns=['Brand'], inplace=True)

X = df.drop('Price', axis=1) 
y = df['Price'] 

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the features (excluding target variable)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression(learning_rate=0.01, n_iterations=1000) 
model.fit(X_train_scaled, y_train)
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

# Results
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print("Training R² Score:", train_score)
print("Testing R² Score:", test_score)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions, alpha=0.5, label='Predictions')
min_val = min(min(y_test), min(test_predictions))
max_val = max(max(y_test), max(test_predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Phone Prices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.margins(0.1)
plt.savefig('price_predictions.png')
plt.show()
