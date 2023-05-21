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
def predictform():
    if request.method == 'POST':
        age = request.form['age']
        bp = request.form['bp']
        sg = request.form['sg']
        rbc = request.form['rbc']
        pcc = request.form['pcc']
        ba = request.form['ba']
        bu = request.form['bu']
        sc = request.form['sc']
        pot = request.form['pot']
        htn = request.form['htn']
        dm = request.form['dm']
        cad = request.form['cad']
        pe = request.form['pe']
        ane = request.form['ane']
        data = convert_to_df(age, bp, sg, rbc, pcc, ba, bu, sc, pot, htn, dm, cad, pe, ane)
        print(data.head())
        try:
            prediction = predict(data)
            if prediction == 1:
                prediction = 'You have Chronic Kidney Disease'
            else:
                prediction = 'You do not have Chronic Kidney Disease'
            return render_template('predict.html', result=prediction)
        except:
            return render_template('predict.html', result="invalid data")
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
 