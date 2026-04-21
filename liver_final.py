import streamlit as st
import joblib
import numpy as np
pip install -r requirements.txt

# Charger modèle
@st.cache_resource
def load_model():
    return joblib.load("model_rf.sav")

model = load_model()

#######
st.title("Liver Disease Classification")

st.subheader("Patient details")
Age = st.number_input("Age", min_value=0)
BMI = st.number_input("BMI", min_value=0.0001)
    
####
st.subheader("Medical history")
mapping = {"No": 0, "Yes": 1}
Comorb_Diabetes = mapping[st.radio("Do you have diabetes?",("No", "Yes"))]
Comorb_Genetic_History = mapping[st.radio("Do you have genetic predisposition for liver diseases?",("No", "Yes"))]

####
st.subheader("Common symptoms related to liver diseases")

mapping = {"No": 0, "Yes": 1}

Sym_Fatigue = mapping[st.radio("Do you experience symptoms of fatigue?",("No", "Yes"))]
Sym_Jaundice = mapping[st.radio("Do you experience symptoms of jaundice?",("No", "Yes"))]
Sym_Abdominal_Pain = mapping[st.radio("Do you experience abdominal pain?",("No", "Yes"))]
Sym_Itching = mapping[st.radio("Do you experience itching?",("No", "Yes"))]
Sym_Ascites = mapping[st.radio("Do you have ascites?",("No", "Yes"))]
Sym_Dark_Urine = mapping[st.radio("Do you have dark urine frequently?",("No", "Yes"))]

#####

st.subheader("Test results for liver health")
ALT = st.number_input("ALT in units/L", min_value=0)
AST = st.number_input("AST in units/L", min_value=0)
Bilirubin = st.number_input("Bilirubin in mg/dL", min_value=0)
Albumin = st.number_input("Albumin in g/dL", min_value=0)
Platelets = st.number_input("Platelets in 10^9/L", min_value=0)
GGT = st.number_input("GGT in units/L", min_value=0)
Triglycerides = st.number_input("Triglycerides in mg/dL", min_value=0)
INR = st.number_input("INR", min_value=0)



if st.button("Predict"):
    features = np.array([[Age, BMI, Sym_Fatigue, Sym_Jaundice,
                          Sym_Abdominal_Pain,Sym_Itching, Sym_Ascites,Sym_Dark_Urine,Comorb_Diabetes,
                          Comorb_Genetic_History, ALT,AST,Bilirubin,
                          Albumin, Platelets, GGT, Triglycerides, INR]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]
    
    liver_disease={0:'Alcoholic Liver Disease',
                   1:'Fatty Liver Disease (NAFLD)',
                   2:'General Liver Disease Severity',
                   3:'Healthy Liver',
                   4:'Liver Cirrhosis Risk'}
    if prediction==3:
        st.success(f"You have a {liver_disease[prediction]} (Probability : {proba:.2%})")
    elif prediction==2:
        st.warning(f"You have a {liver_disease[prediction]} (Probability : {proba:.2%})")
    else:
        st.error(f"You have a {liver_disease[prediction]} (Probability : {proba:.2%})")
