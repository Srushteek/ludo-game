import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict fraud
def predict_fraud():
    try:
        # Retrieve and scale input values
        features = [float(entry.get()) for entry in entries]
        features = np.array(features).reshape(1, -1)
	#if features.shape[1] != 29:            
	   # raise ValueError("Incorrect number of features #provided.")
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

        # Display result
        messagebox.showinfo('Prediction', f'Transaction is: {result}')
    except ValueError as ve:
        messagebox.showerror('Error', f'Invalid input: {ve}')
    except Exception as e:
        messagebox.showerror('Error', str(e))

# Create the main window
root = tk.Tk()
root.title('Credit Card Fraud Detection')

# Create and place labels and entries for input features
labels = [f'Feature {i+1}' for i in range(29)]  # Adjust if necessary
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)

# Create and place Predict button
predict_button = tk.Button(root, text='Predict', command=predict_fraud)
predict_button.grid(row=len(labels), columnspan=2)

# Run the GUI event loop
root.mainloop()
