import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example input values for 29 features
features = [
    0.5, -0.3, 0.4, -0.2, 0.6, -0.4, 0.3, -0.5, 0.7, -0.6,
    0.8, -0.7, 0.9, -0.8, 1.0, -0.9, 0.2, -0.1, 0.4, -0.3,
    0.5, -0.4, 0.6, -0.5, 0.7, -0.6, 0.8, -0.7, 10000.0, 3600
]

# Convert to numpy array and scale
features = np.array(features).reshape(1, -1)
features_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(features_scaled)
result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

print(f'Transaction is: {result}')
