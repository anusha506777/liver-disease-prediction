# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('indian_liver_patient.csv')

# Rename unnamed column if exists
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Preprocessing
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Male=1, Female=0
df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)
df['Dataset'] = df['Dataset'].replace(2, 0)  # 1 = Liver Disease, 0 = No Disease

X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('liver_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as liver_model.pkl")
