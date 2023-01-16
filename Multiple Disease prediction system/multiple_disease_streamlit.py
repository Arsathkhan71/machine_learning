# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 23:24:35 2023

@author: Arsath khan
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd

#loading model
model_cancer = pickle.load(open('Multiple Disease prediction system/breast_cancer/breast_cancer.pkl', 'rb'))
model_diabetes = pickle.load(open('Multiple Disease prediction system/diabetes/diabetes.pkl', 'rb'))
model_heart = pickle.load(open('Multiple Disease prediction system/heart_disease/heart_disease.pkl', 'rb'))
#sidebar for navigation
with st.sidebar:
    selected = option_menu("Multiple Prediction System",
                           ['Heart Disease Prediction',
                            'Diabetes Prediction',
                            'Breast Cancer Prediction'],
                           icons=['heart','activity','person'],
                           menu_icon = ['cast'],
#                            orientation = 'vertical',
#                            styles = {
#                                'container' : {"padding": "3!important", "background-color": "#fafafa"},
#                                "nav-link": {"font-size": "15px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"}

#                            },
                           default_index = 0 )



#breast cancer prediction    
if(selected == 'Breast Cancer Prediction'):
    
    #title
    st.title('Breast Cancer Prediction')

    #Arranging input area in grid format
    col1, col2, col3, col4 = st.columns(4)

    #column names
    # ['worst concave points','worst perimeter','mean concave points','worst radius',
    # 'mean perimeter','worst area','mean radius','mean area',
    # 'mean concavity','worst concavity','mean compactness','worst compactness',
    # 'radius error','perimeter error','area error','worst texture',
    # 'worst smoothness','worst symmetry','mean texture']

    with col1:
        worst_concave_points = st.text_input("Worst concave points")
        mean_perimeter = st.text_input("Mean Perimeter")
        mean_concavity = st.text_input("Mean Concavity")
        radius_error = st.text_input("Radius Error")
        worst_smoothness = st.text_input("Worst Smoothness")

    with col2:
        worst_perimeter = st.text_input("Worst Perimeter")
        worst_area = st.text_input("Worst Area")
        worst_concavity = st.text_input("Worst Concavity")
        perimeter_area = st.text_input("Perimeter Area")
        worst_symmetry = st.text_input('Worst Symmetry')

    with col3:
        mean_concave_points = st.text_input("Mean_Concave_Points")
        mean_radius = st.text_input("Mean Radius")
        mean_compactness = st.text_input("Mean Compactness")
        area_error = st.text_input("Area Error")
        mean_texture = st.text_input("Mean Texture")

    with col4:
        worst_radius = st.text_input("Worst Radius")
        mean_area = st.text_input("Mean Area")
        worst_compactness = st.text_input("Worst Compactness")
        worst_texture = st.text_input("Worst Texture")

    data = [worst_concave_points,worst_perimeter,mean_concave_points,worst_radius,
            mean_perimeter,worst_area,mean_radius,mean_area,mean_concavity,
            worst_concavity,mean_compactness,worst_compactness,radius_error,
            perimeter_area,area_error,worst_texture,worst_smoothness,worst_symmetry,
            mean_texture]

    arr_ip = np.array([data])

    df = pd.DataFrame(data=arr_ip, columns=['worst concave points','worst perimeter','mean concave points','worst radius','mean perimeter','worst area','mean radius','mean area','mean concavity','worst concavity','mean compactness','worst compactness','radius error','perimeter error','area error','worst texture','worst smoothness','worst symmetry','mean texture'])

    breast_cancer_pred = ""

    if (st.button("Predict")):
        breast_cancer_model = model_cancer.predict(df)
        if (breast_cancer_model[0] == 0):
            breast_cancer_pred = "Malignant"
        else:
            breast_cancer_pred = 'Benign'
    st.success(breast_cancer_pred)

    
#diabetes prediction
if (selected == 'Diabetes Prediction'):
    
    #title
    st.title("Diabetes Prediction")
    
    col1,col2,col3 = st.columns(3)
    
    #columns
#     ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
    
    with col1:
        pregnancies = st.text_input("Pregnancies")
        skinthickness = st.text_input("SkinThickness")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
        
    with col2:
        glucose = st.text_input("Glucose")
        insulin = st.text_input("Insulin")
        age = st.text_input("Age")
       
    with col3:
        blood_pressure = st.text_input("Blood Pressure")
        bmi = st.text_input("BMI")
        
    data = [pregnancies,glucose,blood_pressure,skinthickness,insulin,bmi,DiabetesPedigreeFunction,age]
    
    arr_ip = np.array([data])
    df = pd.DataFrame(data=arr_ip, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'])
                      
    diabetes_pred = ""
    
    if(st.button("Predict")):
        
       diabetes_model = model_diabetes.predict(df)
       if(diabetes_model[0] == 1):
            diabetes_pred = "Person is Diabetic"
       else:
            diabetes_pred = "Person in Non-Diabetic"
    st.success(diabetes_pred)
    
#heart disease prediction
if (selected == 'Heart Disease Prediction'):
    
    #title
    st.title("Heart Disease Prediction")
    
    col1,col2,col3 = st.columns(3)
    
#     ['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang','oldpeak', 'slope', 'ca', 'thal']
    
    with col1:
        age = st.text_input("Age")
        trestbps = st.text_input("Blood Pressure")
        thalach = st.text_input("Max Heart Rate")
        slope = st.selectbox("Slope",options=(0,1,2))
        
    with col2:
#         sex = st.radio("Gender",options=('Male','Female'),horizontal=
        sex = st.selectbox("Gender",options=('Male','Female'))
        chol = st.text_input("Cholestrol")
        exang = st.selectbox("Exercise Angina", options=('Yes','No'))
        ca = st.selectbox("No of Major Vessels",options=(0,1,2,3,4))
    with col3:
        cp = st.selectbox("Chest Pain Type",options=(0,1,2,3))
        restecg = st.selectbox("Resting ECG",options=(0,1,2))
        oldpeak = st.text_input("Oldpeak")
        thal = st.selectbox("Thal",options=(0,1,2,3))
        
    data = [age,sex,cp,trestbps,chol,restecg,thalach,exang,oldpeak,slope,ca,thal]
    
    arr_ip = np.array([data])
    
    df = pd.DataFrame(data=arr_ip,columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang','oldpeak', 'slope', 'ca', 'thal'])
    
    df['sex'] = df['sex'].map({'Male':1, "Female":0})
    df['exang'] = df['exang'].map({'Yes':1,"No":0})
    
    heart_pred = ""
                      
    if (st.button("Predict")):
        heart_model = model_heart.predict(df)
        if heart_model[0] == 1:
            heart_pred = "Person Have Heart Disease"
        else:
            heart_pred = "Person Doesn't Have Heart Disease"
    st.success(heart_pred)