
# liver_disease_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("liver.csv")  # Ensure the file is in the same directory

# Preprocessing
df = df.dropna()  # Remove missing values

# Encode categorical variables (if any)
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Define features and target
X = df.drop(['Dataset'], axis=1)  # Replace 'Dataset' with the actual target column name if different
y = df['Dataset']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'liver_model.pkl')
print("Model saved as 'liver_model.pkl'")
