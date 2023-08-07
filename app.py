import streamlit as st
import numpy as np
import pickle

sc = pickle.load(open('sc.pkl', 'rb'))
model = pickle.load(open('classifier.pkl', 'rb'))

def preprocess_data(features):
    scaled_features = sc.transform([features])
    return scaled_features

def predict_diabetes(features):
    scaled_features = preprocess_data(features)
    prediction = model.predict(scaled_features)
    return prediction[0]

def main():
    st.title("Diabetes Prediction App")
    st.write("Enter the following information to predict diabetes:")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=250, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=22.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=150, value=30)

    if st.button("Predict"):
        features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        prediction = predict_diabetes(features)

        if prediction == 1:
            st.write("You have diabetes.")
        else:
            st.write("You don't have diabetes.")

if __name__ == "__main__":
    main()
