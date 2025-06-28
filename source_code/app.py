# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('liver_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['Age'])
        gender = int(request.form['Gender'])  # Male=1, Female=0
        total_bilirubin = float(request.form['Total_Bilirubin'])
        direct_bilirubin = float(request.form['Direct_Bilirubin'])
        alk_phosphate = float(request.form['Alkaline_Phosphotase'])
        alamine = float(request.form['Alamine_Aminotransferase'])
        aspartate = float(request.form['Aspartate_Aminotransferase'])
        total_protein = float(request.form['Total_Protiens'])
        albumin = float(request.form['Albumin'])
        ratio = float(request.form['Albumin_and_Globulin_Ratio'])

        features = np.array([[age, gender, total_bilirubin, direct_bilirubin,
                              alk_phosphate, alamine, aspartate,
                              total_protein, albumin, ratio]])
        prediction = model.predict(features)

        result = "Liver Disease" if prediction[0] == 1 else "No Liver Disease"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
