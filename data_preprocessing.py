import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
# Load the dataset
data = pd.read_csv(r"C:\Users\Lenovo\Downloads\creditcard (1).csv")

# Drop unnecessary columns
X = data.drop(['Time', 'Class'], axis=1)
print(X.shape)  
#y = data['Class']

# Feature scaling
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
X_train, X_test, y_train, y_test = train_test_split(X, data['Class'], test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_resampled, y_train_resampled)

# Save the model and scaler
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')