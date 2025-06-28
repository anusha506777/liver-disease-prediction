# 🩺 Liver Disease Prediction using Machine Learning

This project predicts liver disease using machine learning algorithms and provides a user-friendly interface using Flask.

---

## 📌 Problem Statement

Early detection of liver disease can prevent serious complications and improve patient outcomes. The goal of this project is to build an ML model that predicts liver disease based on patient data and integrate it into a web application.

---

## 🚀 Project Workflow

1. **Data Collection**: Used Indian Liver Patient Dataset (ILPD).
2. **Data Preprocessing**: 
   - Handled missing values
   - Label encoding of categorical data
   - Feature scaling
3. **EDA**: Used visualization to understand data distribution and correlation.
4. **Model Training**:
   - Trained multiple models (RandomForest, Logistic Regression, etc.)
   - Selected best performing model
5. **Model Evaluation**: Evaluated using accuracy and confusion matrix.
6. **Flask App**: Created a web interface for predictions.

---

## 🧠 Algorithms Used

- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors
- Decision Tree Classifier

---

## 💻 Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn
- Flask (for web framework)
- HTML/CSS/Bootstrap (for frontend)

---

## 📂 Project Structure
liver_prediction_project/
│
├── Notebook/
│ └── liver_disease_prediction(EDA).ipynb
├── source_code/
│ └── liver_disease_prediction.py
├── train_model.py
├── app.py
├── liver_model.pkl
├── templates/
│ └── index.html
├── static/
│ └── style.css
└── indian_liver_patient.csv
