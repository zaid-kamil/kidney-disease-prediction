from flask import Flask, render_template, redirect, request, flash
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)
app.secret_key = "09klsa asdjk"
model_names = ['Logistic Regression',
                'Decision Tree',
                'Random Forest',
                'AdaBoost',
                'Gradient Boosting',
                'SVC',
                'KNN',
                'GaussianNB',
                'Voting Classifier']
# load models
def load_models():
    models = []
    for model in model_names:
        models.append(joblib.load(f'{model}.pkl'))
    return models

# predict function
def predict(data):
    models = load_models()
    results = []
    for model in models:
        results.append(model.predict(data)[0])  
    # return most frequent prediction
    most_frequent = np.bincount(results).argmax()
    return most_frequent

def convert_to_df(age, bp, sg, rbc, pcc, ba, bu, sc, pot, htn, dm, cad, pe, ane):
    data = {
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'rbc': [rbc],
        'pcc': [pcc],
        'ba': [ba],
        'bu': [bu],
        'sc': [sc],
        'pot': [pot],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'pe': [pe],
        'ane': [ane]
    }
    return pd.DataFrame(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        bp = int(request.form['bp'])
        sg = float(request.form['sg'])
        rbc = int(request.form['rbc'])
        pcc = int(request.form['pcc'])
        ba = int(request.form['ba'])
        bu = int(request.form['bu'])
        sc = float(request.form['sc'])
        pot = float(request.form['pot'])
        htn = int(request.form['htn'])
        dm = int(request.form['dm'])
        cad = int(request.form['cad'])
        pe = int(request.form['pe'])
        ane = int(request.form['ane'])
        data = convert_to_df(age, bp, sg, rbc, pcc, ba, bu, sc, pot, htn, dm, cad, pe, ane)
        prediction = predict(data)
        if prediction == 1:
            prediction = 'You have Chronic Kidney Disease'
        else:
            prediction = 'You do not have Chronic Kidney Disease'
        return render_template('predict.html', prediction=prediction)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
 