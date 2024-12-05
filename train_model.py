import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Load your dataset
data = pd.read_csv("inventory_data.csv")  # Replace with your dataset file name

# Feature engineering
data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Week'] = data['Date'].dt.day_name()  # Extract day of the week

# Convert categorical variables to one-hot encoding
data = pd.get_dummies(data, columns=['Item', 'Day_of_Week', 'Weather'], drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['Sales', 'Date'])  # Replace 'Sales' with your target column
y = data['Sales']

# Debugging: Check variability in target and features
print("Target variable (y) unique values:", y.unique())
print("Feature variability (X):")
print(X.describe())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the feature names to ensure consistency during prediction
with open("feature_names.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

# Train the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("demand_forecast_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as 'demand_forecast_model.pkl'")
print("Feature names saved as 'feature_names.pkl'")

# Test the model with a sample input
sample_input = pd.DataFrame([X_train.iloc[0]], columns=X_train.columns)  # Retain feature names
predicted_value = model.predict(sample_input)
print("Sample prediction:", predicted_value)
