import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from io import BytesIO
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import streamlit.components.v1 as components

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="GlucoTrack", page_icon="🩺", layout="wide")

# ==============================
# LOAD MODEL
# ==============================
model = pickle.load(open("diabetes_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ==============================
# SESSION STATE
# ==============================
defaults = {
    "started": False,
    "logged_in": False,
    "role": None,
    "page": "User Login",
    "result_ready": False,
    "users": {"user@gmail.com": "1234"},
    "admins": {"admin@glucotrack.com": "admin@123"},
    "prediction_done": False,
    "patient_data": None,
    "prediction_result": None,
    "confidence": None,
    "pdf_bytes": None
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================
# HOME PAGE
# ==============================
def home_page():
    st.markdown("""
    <div class="hero">
        <h1>🩺 GlucoTrack</h1>
        <h3>AI Powered Diabetes Risk Prediction & Smart Health Analytics</h3>
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    with col1:
        st.info("📊 ML-Based Diabetes Prediction")
        st.write("Predict diabetes risk using advanced ML model trained on health data.")

    with col2:
        st.success("📈 Patient Health Analytics")
        st.write("Visual dashboards to understand health parameters easily.")

    with col3:
        st.warning("💡 Personalized Health Suggestions")
        st.write("Get lifestyle and diet recommendations based on health data.")

    if st.button("🚀 Get Started"):
        st.session_state.started = True
        st.rerun()

if not st.session_state.started:
    home_page()
    st.stop()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🩺 GLUCOTRACK")

menu_options = ["User Login","Sign Up","Admin Login"]
if st.session_state.logged_in:
    menu_options = ["Prediction","Result Dashboard"]
    if st.session_state.role=="Admin":
        menu_options.insert(0,"Admin Dashboard")

st.session_state.page = st.sidebar.radio("",menu_options)

if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in=False
        st.session_state.role=None
        st.session_state.page="User Login"
        st.rerun()

# ==============================
# LOGIN
# ==============================
def user_login():
    st.title("User Login")
    email=st.text_input("Email")
    pwd=st.text_input("Password",type="password")
    if st.button("Login"):
        if email in st.session_state.users and st.session_state.users[email]==pwd:
            st.session_state.logged_in=True
            st.session_state.role="User"
            st.session_state.page="Prediction"
            st.rerun()
        else:
            st.error("Invalid credentials")

def signup():
    st.title("Sign Up")
    email=st.text_input("Email")
    pwd=st.text_input("Password",type="password")
    if st.button("Create"):
        st.session_state.users[email]=pwd
        st.success("Account created")

def admin_login():
    st.title("Admin Login")
    email=st.text_input("Admin Email")
    pwd=st.text_input("Password",type="password")
    if st.button("Login"):
        if email in st.session_state.admins and st.session_state.admins[email]==pwd:
            st.session_state.logged_in=True
            st.session_state.role="Admin"
            st.session_state.page="Admin Dashboard"
            st.rerun()

def admin_dashboard():
    st.title("Admin Dashboard")
    st.metric("Users",len(st.session_state.users))

# ==============================
# PDF
# ==============================
def generate_pdf(patient_data,result,confidence):
    buffer=BytesIO()
    pdf=canvas.Canvas(buffer,pagesize=A4)
    pdf.drawString(50,800,"GlucoTrack Report")
    pdf.drawString(50,780,f"Result: {result}")
    pdf.drawString(50,760,f"Confidence: {confidence}%")
    y=730
    for k,v in patient_data.items():
        pdf.drawString(50,y,f"{k}: {v}")
        y-=20
    pdf.save()
    return buffer.getvalue()

# ==============================
# PREDICTION PAGE
# ==============================
def prediction_page():
    st.title("Diabetes Prediction")

    col1,col2=st.columns(2)
    with col1:
        preg=st.number_input("Pregnancies",0,20)
        glucose=st.number_input("Glucose",50,250)
        bp=st.number_input("Blood Pressure",30,140)
    with col2:
        insulin=st.number_input("Insulin",0,400)
        bmi=st.number_input("BMI",10.0,70.0)
        age=st.number_input("Age",1,100)

    if st.button("Predict Diabetes Risk"):
        data={"Pregnancies":preg,"Glucose":glucose,"BloodPressure":bp,"Insulin":insulin,"BMI":bmi,"Age":age}
        df=pd.DataFrame([data])
        df=df.reindex(columns=columns,fill_value=0)

        pred=model.predict(df)
        prob=model.predict_proba(df)[0]

        if pred[0]==1:
            result="High Risk of Diabetes"
            confidence=round(prob[1]*100,2)
        else:
            result="Low Risk of Diabetes"
            confidence=round(prob[0]*100,2)

        st.session_state.patient_data=data
        st.session_state.prediction_result=result
        st.session_state.confidence=confidence
        st.session_state.pdf_bytes=generate_pdf(data,result,confidence)
        st.session_state.result_ready=True
        st.session_state.page="Result Dashboard"
        st.rerun()

# ==============================
# RESULT DASHBOARD PAGE
# ==============================
def result_dashboard():
    st.title("Prediction Result Dashboard")

    if not st.session_state.result_ready:
        st.warning("No prediction yet")
        return

    st.success(st.session_state.prediction_result)
    st.metric("Confidence",f"{st.session_state.confidence}%")

    st.bar_chart(pd.DataFrame([st.session_state.patient_data]))

    st.download_button(
        "Download Report",
        st.session_state.pdf_bytes,
        "report.pdf"
    )

# ==============================
# ROUTING
# ==============================
page=st.session_state.page

if page=="User Login": user_login()
elif page=="Sign Up": signup()
elif page=="Admin Login": admin_login()
elif page=="Admin Dashboard": admin_dashboard()
elif page=="Prediction": prediction_page()
elif page=="Result Dashboard": result_dashboard()
