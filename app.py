# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:05:40 2020

@author: gaurav sahani
"""


from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            Type = float(request.form['type'])
            FixedAcidity = float(request.form['fixed_acidity'])
            VolatileAcidity = float(request.form['volatile_acidity'])
            CitricAcid = float(request.form['citric_acid'])
            ResidualSugar = float(request.form['residual_sugar'])
            Chlorides = float(request.form['chlorides'])
            FreeSulfurdioxide = float(request.form['free_sulfur_dioxide'])
            TotalSulfurdioxide = float(request.form['total_sulfur_dioxide'])
            Density = float(request.form['density'])
            pH = float(request.form['pH'])
            Sulphates = float(request.form['sulphates'])
            Alcohol = float(request.form['alcohol'])
            pred_args = [Type, FixedAcidity, VolatileAcidity, CitricAcid, ResidualSugar, Chlorides,
                         FreeSulfurdioxide,	TotalSulfurdioxide,	Density, pH, Sulphates,	Alcohol]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            Wine_Quality_Model = open("catboost_classification_model.pkl", "rb")
            ml_model = joblib.load(Wine_Quality_Model)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = float(model_prediction)
        
        except ValueError:
            return "Please Check if values are written correctly"
    return render_template('predict.html', prediction=model_prediction)

if __name__ == "__main__":
    app.run(debug=True)