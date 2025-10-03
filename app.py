import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# import ridge regressor and standard scaler pickle files
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictData", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data
        features = [
            float(request.form.get('Temperature')),
            float(request.form.get('RH')),
            float(request.form.get('Ws')),
            float(request.form.get('Rain')),
            float(request.form.get('FFMC')),
            float(request.form.get('DMC')),
            float(request.form.get('ISI')),
            float(request.form.get('Classes')),
            float(request.form.get('Region'))
        ]
        
        # Convert the list to a 2D array (required for the scaler's transform method)
        features_array = [features]  # This turns the list into a 2D array
        
        # Apply scaling transformation
        new_data_data=scaler_model.transform(features_array)
        result=ridge_model.predict(new_data_data)
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0')