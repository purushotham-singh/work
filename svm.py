import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Simulated dataset (Replace with actual extracted features)
data = {
    "T0": [0, 0, 0, 0],
    "T1": [150, 180, 220, 250],
    "TOF": [150, 180, 220, 250],
    "Tp": [310, 320, 330, 340],
    "Tr": [450, 460, 470, 480],
    "FFT_Magnitude": [80, 90, 110, 120],
    "Ap": [1.5, 1.6, 1.8, 2.0],
    "Ar": [0.05, 0.07, 0.09, 0.12],
    "Label": [0, 0, 1, 1]  # 0 = Good, 1 = Defective
}

df = pd.DataFrame(data)

# Features and Labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Ensure at least two distinct classes exist
if len(np.unique(y)) < 2:
    raise ValueError("SVM needs at least two distinct classes for classification. Check your labels.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = svm_model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
