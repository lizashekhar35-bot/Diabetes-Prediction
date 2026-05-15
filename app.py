import streamlit as st
import pandas as pd
import pickle

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="MediMate",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# LOAD MODEL & COLUMNS
# ==========================================
model = pickle.load(open("diabetes_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ==========================================
# SESSION STATE
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>
    🩺 MediMate
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ======================================
    # DARK / LIGHT MODE
    # ======================================
    st.markdown("## 🎨 Theme Mode")

    theme = st.radio(
        "Choose Mode",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1
    )

    st.session_state.theme = theme

    st.markdown("---")

    # ======================================
    # LOGIN / SIGNUP / ADMIN
    # ======================================
    st.markdown("## 🔐 Menu")

    st.button("🔑 Login")
    st.button("📝 Signup")
    st.button("👨‍💻 Admin")

    st.markdown("---")

    st.info("""
    👨‍💻 Admin Credentials

    Username: admin  
    Password: 1234
    """)

# ==========================================


# ==========================================
# LIGHT MODE CSS
# ==========================================
else:# ======================================
# DARK MODE CSS
# ======================================
if st.session_state.theme == "Dark":

    st.markdown("""
    <style>

    /* MAIN APP */
    .stApp {
        background-color: #020817;
        color: white;
    }

    /* SIDEBAR BACKGROUND */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
    }

    /* SIDEBAR TEXT */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* INPUT LABELS */
    label {
        color: white !important;
    }

    /* TEXT INPUT */
    .stTextInput input {
        background-color: #1E293B;
        color: white;
        border-radius: 10px;
    }

    /* NUMBER INPUT */
    .stNumberInput input {
        background-color: #1E293B;
        color: white;
    }

    /* BUTTONS */
    div.stButton > button {
        background-color: #1E293B;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        border: none;
        font-size: 16px;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #334155;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>

    div.stButton > button {
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    </style>
    """, unsafe_allow_html=True)

# ==========================================
# SIGN IN PAGE FIRST
# ==========================================
if not st.session_state.logged_in:

    st.markdown("""
    <h1 style='text-align:center;'>
    🔐 Sign In to MediMate
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("###")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    st.markdown("###")

    if st.button("🚀 Sign In"):

        if username == "admin" and password == "1234":

            st.session_state.logged_in = True

            st.success("✅ Login Successful")

            st.rerun()

        else:

            st.error("❌ Invalid Username or Password")

    st.stop()

# ==========================================
# MAIN APP
# ==========================================
st.markdown("""
<h1 style='text-align:center;'>
🩺 Diabetes Prediction App
</h1>
""", unsafe_allow_html=True)

st.write("### Enter Patient Details")

st.markdown("---")

# ==========================================
# LOGOUT BUTTON
# ==========================================
if st.button("🚪 Logout"):

    st.session_state.logged_in = False
    st.rerun()

# ==========================================
# INPUTS
# ==========================================
col1, col2 = st.columns(2)

with col1:

    preg = st.number_input("Pregnancies", 0, 20, 1)

    glucose = st.number_input("Glucose", 50, 200, 120)

    bp = st.number_input("Blood Pressure", 30, 120, 70)

    skin = st.number_input("Skin Thickness", 0, 100, 20)

with col2:

    insulin = st.number_input("Insulin", 0, 300, 100)

    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        0.0,
        3.0,
        0.5
    )

    age = st.number_input("Age", 1, 100, 30)

st.markdown("###")

# ==========================================
# PREDICTION
# ==========================================
if st.button("🔍 Predict"):

    # ======================================
    # CREATE DATAFRAME
    # ======================================
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

    # ======================================
    # FEATURE ENGINEERING
    # ======================================
    input_raw['Glucose_BMI'] = (
        input_raw['Glucose'] * input_raw['BMI']
    )

    input_raw['Insulin_Glucose'] = (
        input_raw['Insulin'] * input_raw['Glucose']
    )

    input_raw['Age_BMI'] = (
        input_raw['Age'] * input_raw['BMI']
    )

    input_raw['BMI_Squared'] = (
        input_raw['BMI'] ** 2
    )

    # ======================================
    # ENCODING
    # ======================================
    input_encoded = pd.get_dummies(input_raw)

    # ======================================
    # MATCH TRAINING COLUMNS
    # ======================================
    input_df = input_encoded.reindex(
        columns=columns,
        fill_value=0
    )

    # ======================================
    # PREDICTION
    # ======================================
    prediction = model.predict(input_df)

    st.markdown("---")

    # ======================================
    # OUTPUT
    # ======================================
    if prediction[0] == 1:

        st.error("""
        ⚠️ High Risk of Diabetes
        """)

    else:

        st.success("""
        ✅ Low Risk of Diabetes
        """)
