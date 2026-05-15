import streamlit as st
import pandas as pd
import pickle

# ==============================
# LOAD MODEL & COLUMNS
# ==============================
model = pickle.load(open("diabetes_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ==============================
# LOGIN PAGE
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Login to MediMate")

    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Show login page if not logged in
if not st.session_state.logged_in:
    login_page()
    st.stop()

# ==============================
# MAIN APP UI
# ==============================
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details below:")

# Logout button
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# Inputs
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 50, 200, 120)
bp = st.number_input("Blood Pressure", 30, 120, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 300, 100)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 30)

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):

    input_raw = pd.DataFrame({
        'Pregnancies': [preg],
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'SkinThickness': [skin],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    input_raw['Glucose_BMI'] = input_raw['Glucose'] * input_raw['BMI']
    input_raw['Insulin_Glucose'] = input_raw['Insulin'] * input_raw['Glucose']
    input_raw['Age_BMI'] = input_raw['Age'] * input_raw['BMI']
    input_raw['BMI_Squared'] = input_raw['BMI'] ** 2

    input_encoded = pd.get_dummies(input_raw)
    input_df = input_encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
