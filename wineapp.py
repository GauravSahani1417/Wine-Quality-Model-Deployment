# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
import pickle

html_temp = """
    <div style="background-color:#010200;padding:10px">
    <h1 style="color:white;text-align:center;">Red Wine Quality</h1>
    </div>
    """    
st.markdown(html_temp,unsafe_allow_html=True)


def user_input_features():
    fixed_acidity = st.slider('FA', 6, 16)
    volatile_acidity = st.slider('VA', 0.25, 1.50)
    citric_acidity = st.slider('CA', 0.0, 1.0)
    residual_sugar = st.slider('RS', 0.0, 7.5)
    chlorides = st.slider('Cl', 0.0, 0.3)
    free_sulfur_dioxide = st.slider('FSD', 0, 60) 
    total_sulfur_dioxide	 = st.slider('TSD', 0, 200)
    density = st.slider('D', 0.9900, 1.0025)
    pH = st.slider('pH', 2.8, 3.8)
    sulphates = st.slider('Su', 0.0, 1.5)
    alcohol = st.slider('Al', 0, 14)
    data = {'FA': fixed_acidity,
            'VA': volatile_acidity,
            'CA': citric_acidity,
            'RS': residual_sugar, 
            'Cl': chlorides,
            'FSD': free_sulfur_dioxide,
            'TSD': total_sulfur_dioxide,
            'D': density,
            'pH': pH,
            'Su': sulphates,
            'Al': alcohol
            } 
    
    features = pd.DataFrame(data, index=[0])
    return features

st.subheader('Please Enter Input Through Slider')
df = user_input_features()     
        
st.subheader('User Input parameters')
st.write(df)

model=pickle.load(open('Wine_GS.pkl','rb'))

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediction Probability')
st.write('The quality of red wine is inferior:',prediction_proba[0][0]*100)
st.write('The quality of red wine is fine:',prediction_proba[0][1]*100)
st.write('The quality of red wine is superior:',prediction_proba[0][2]*100)


html_temp1 = """
    <div style="background-color:#010200">
    <p style="color:white;text-align:center;" >Designe & Developed By: <b>Tanvi Kurade</b> </p>
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)