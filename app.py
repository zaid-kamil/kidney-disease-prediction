from flask import Flask, render_template, redirect, request, flash
import numpy as np
import joblib


app = Flask(__name__)
app.secret_key = "09klsa asdjk"
model_names = [
    
]
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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
 