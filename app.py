# Streamlit Diabetes Prediction Web App (Modern UI + Login Page)

```python
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# ---------------- DARK / LIGHT MODE ----------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    mode = st.toggle("🌙 Dark Mode", value=True)

    if mode:
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

    st.markdown("---")
    menu = st.radio(
        "Navigation",
        ["Login", "Signup", "Admin"]
    )

# ---------------- CSS ----------------
if st.session_state.theme == 'dark':
    bg_color = '#0f172a'
    card = '#1e293b'
    text = 'white'
    sidebar = '#020617'
else:
    bg_color = '#f8fafc'
    card = '#ffffff'
    text = '#111827'
    sidebar = '#e2e8f0'

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text};
}}

section[data-testid="stSidebar"] {{
    background-color: {sidebar};
}}

.main-card {{
    background-color: {card};
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
}}

.title {{
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #38bdf8;
}}

.sub {{
    text-align: center;
    font-size: 18px;
    color: gray;
}}

.stButton>button {{
    width: 100%;
    border-radius: 12px;
    height: 45px;
    font-size: 18px;
    background-color: #2563eb;
    color: white;
    border: none;
}}

.stButton>button:hover {{
    background-color: #1d4ed8;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- SAMPLE DATASET ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

try:
    df = load_data()
except:
    st.warning("Please keep diabetes.csv file in the same folder")
    st.stop()

# ---------------- MODEL TRAINING ----------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# ---------------- LOGIN PAGE ----------------
if menu == "Login":

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>🩺 Diabetes Prediction</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub'>AI Based Health Prediction System</div>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login = st.button("Login")

        if login:
            if username == "admin" and password == "1234":
                st.success("Login Successful")
                st.balloons()

                st.markdown("---")
                st.subheader("📊 Enter Patient Details")

                pregnancies = st.number_input("Pregnancies", 0, 20)
                glucose = st.number_input("Glucose", 0, 200)
                bp = st.number_input("Blood Pressure", 0, 150)
                skin = st.number_input("Skin Thickness", 0, 100)
                insulin = st.number_input("Insulin", 0, 900)
                bmi = st.number_input("BMI", 0.0, 70.0)
                dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
                age = st.number_input("Age", 1, 120)

                if st.button("Predict Diabetes"):
                    input_data = np.array([[pregnancies, glucose, bp, skin,
                                            insulin, bmi, dpf, age]])

                    prediction = model.predict(input_data)

                    if prediction[0] == 1:
                        st.error("⚠️ Person is Diabetic")
                    else:
                        st.success("✅ Person is Not Diabetic")

            else:
                st.error("Invalid Username or Password")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- SIGNUP PAGE ----------------
elif menu == "Signup":

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📝 Signup")

    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")

    if st.button("Create Account"):
        st.success("Account Created Successfully")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ADMIN PAGE ----------------
else:

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📊 Admin Dashboard")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Shape")
    st.write(df.shape)

    st.write("### Model Accuracy")
    st.success(f"Accuracy: {round(acc*100,2)}%")

    st.write("### Statistical Summary")
    st.dataframe(df.describe())

    st.markdown("</div>", unsafe_allow_html=True)

