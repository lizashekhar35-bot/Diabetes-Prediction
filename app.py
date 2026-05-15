import streamlit as st
import pandas as pd
import pickle

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="MediMate",
    page_icon="🩺",
    layout="centered"
)

# ======================================
# LOAD MODEL & COLUMNS
# ======================================
model = pickle.load(open("diabetes_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ======================================
# SESSION STATE
# ======================================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# ======================================
# SIDEBAR
# ======================================
with st.sidebar:

    st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>
    🩺 MediMate
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================
    # DARK / LIGHT MODE
    # ==========================
    st.markdown("## 🎨 Theme")

    theme = st.radio(
        "Select Mode",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1
    )

    st.session_state.theme = theme

    st.markdown("---")

    # ==========================
    # LOGIN / SIGNUP
    # ==========================
    st.markdown("## 🔐 Account")

    st.text_input("👤 Username")

    st.text_input("🔑 Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        st.button("Login")

    with col2:
        st.button("Signup")

    st.markdown("---")

    # ==========================
    # ADMIN DETAILS
    # ==========================
    st.markdown("## 👨‍💻 Admin")

    st.info("""
    **Username:** admin  
    
    **Password:** 1234
    """)

    st.markdown("---")

    # ==========================
    # APP FEATURES
    # ==========================
    st.markdown("## 📌 Features")

    st.write("✅ Diabetes Prediction")
    st.write("✅ Machine Learning")
    st.write("✅ Light/Dark Mode")
    st.write("✅ User Friendly UI")

# ======================================
# DARK MODE CSS
# ======================================
if st.session_state.theme == "Dark":

    st.markdown("""
    <style>

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }

    .stNumberInput label {
        color: white !important;
    }

    div.stButton > button {
        background-color: #262730;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    div.stButton > button:hover {
        background-color: #3A3B3C;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)

# ======================================
# MAIN TITLE
# ======================================
st.markdown("""
<h1 style='text-align:center;'>
🩺 Diabetes Prediction App
</h1>
""", unsafe_allow_html=True)

st.write("### Enter patient details below:")

st.markdown("---")

# ======================================
# INPUTS
# ======================================
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 50, 200, 120)
    bp = st.number_input("Blood Pressure", 30, 120, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 300, 100)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 100, 30)

st.markdown("###")

# ======================================
# PREDICTION
# ======================================
if st.button("🔍 Predict"):

    # --------------------------
    # Create Input DataFrame
    # --------------------------
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

    # --------------------------
    # Feature Engineering
    # --------------------------
    input_raw['Glucose_BMI'] = input_raw['Glucose'] * input_raw['BMI']

    input_raw['Insulin_Glucose'] = (
        input_raw['Insulin'] * input_raw['Glucose']
    )

    input_raw['Age_BMI'] = (
        input_raw['Age'] * input_raw['BMI']
    )

    input_raw['BMI_Squared'] = (
        input_raw['BMI'] ** 2
    )

    # --------------------------
    # Encoding
    # --------------------------
    input_encoded = pd.get_dummies(input_raw)

    # --------------------------
    # Match Training Columns
    # --------------------------
    input_df = input_encoded.reindex(
        columns=columns,
        fill_value=0
    )

    # --------------------------
    # Prediction
    # --------------------------
    prediction = model.predict(input_df)

    st.markdown("---")

    # --------------------------
    # Output
    # --------------------------
    if prediction[0] == 1:

        st.error("""
        ⚠️ High Risk of Diabetes
        
        Please consult a doctor.
        """)

    else:

        st.success("""
        ✅ Low Risk of Diabetes
        
        Patient appears healthy.
        """)
