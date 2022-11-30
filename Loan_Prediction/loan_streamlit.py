# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:27:15 2022

@author: Arsath khan
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

#load the model

model = pickle.load(open("C:/Users/Arsath khan/Loan_Prediction/loan_model.sav", 'rb'))

#Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Credit_History,Property_Area
st.title('Loan Prediction Model')

Gender = st.radio("Gender",options=("Male","Female"),horizontal=True)
Married = st.radio("Married",options=("Yes","No"),horizontal=True)
Dependents = st.radio("Dependents",options=(0,1),horizontal=True)
Education = st.radio("Education",options=("Graduate","Not Graduate"),horizontal=True)
Self_Employed = st.radio("Self Employed",options=("Yes","No"),horizontal=True)
ApplicantIncome = st.slider("Applicant Income",min_value=(0),max_value=(100000))
CoapplicantIncome = st.slider("Co-Applicant Income",min_value=(0),max_value=(100000))
LoanAmount = st.text_input("Loan Amount")
Credit_History = st.radio('Credit History',options=(0,1),horizontal=True)
Property_Area = st.radio("Property Area",options=('Urban','Rural','Semiurban'),horizontal=True)

data = [Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Credit_History,Property_Area]

#arr_ip = np.array([Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Credit_History,Property_Area])

arr_ip = np.array([data])
print(arr_ip.shape)

#arr_ip_re = arr_ip.reshape(1,10)
#print(arr_ip_re)
pred_df = pd.DataFrame(data=arr_ip,columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Credit_History','Property_Area'])

pred_df['Gender'] = pred_df['Gender'].map({'Male' : 1, 'Female' : 0})
pred_df['Married'] = pred_df['Married'].map({'Yes':1,'No':0})
pred_df['Dependents'] = pred_df['Dependents'].map({'0' : 0, '1' : 1,'2' : 2,'3+' : 3}).astype(int)
pred_df['Education'] = pred_df['Education'].map({'Graduate':1, 'Not Graduate': 0}).astype(int)
pred_df['Self_Employed'] = pred_df['Self_Employed'].map({'Yes': 0, 'No': 1}).astype(int)
pred_df['Property_Area'] = pred_df['Property_Area'].map({'Urban':0, 'Rural':1, 'Semiurban':2}).astype(int)

print(pred_df)

loan_pred = ""

if (st.button("Predict Loan Status")):
    loan_model = model.predict(pred_df)
    
    if (loan_model[0] == 0):
        loan_pred = "Loan is Not Approved"
    else:
        loan_pred = "Loan is Approved"
st.success(loan_pred)


