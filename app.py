import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
@st.cache_data
def load_data():
    # Ensure 'heart_disease.csv' is in the same directory as this script
    data = pd.read_csv('heart_disease.csv')
    return data

# Train the model
@st.cache_resource
def train_model(data):
    # Use only the three selected variables
    X = data[['thalach', 'age', 'cp']]  # Features
    y = data['target']  # Target variable ('1' = disease present, '0' = no disease)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Streamlit app setup
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Add custom styling for a modern look
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f9f9f9;  /* Light grey background for modern feel */
        }
        .stApp {
            background: #f2f2f2;  /* Light background */
        }
        .header {
            font-size: 36px;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 18px;
            color: #34495E;
            text-align: center;
            margin-bottom: 30px;
        }
        .input-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .predict-button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
        }
        .result-success {
            color: #27AE60;
            font-size: 20px;
            text-align: center;
        }
        .result-danger {
            color: #E74C3C;
            font-size: 20px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header for the app
st.markdown("<div class='header'>Heart Disease Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Assess your heart health with this simple tool</div>", unsafe_allow_html=True)

# Load dataset and train model
data = load_data()
model = train_model(data)

# Layout for the input fields and image
col1, col2 = st.columns([1, 2])

# Add an image to the first column
with col1:
    st.image("heart_image.png", caption="Heart Health Awareness", use_container_width=True)

# Add input fields in the second column with modern container
with col2:
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)

    st.markdown("### Enter Your Health Details")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    cp = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)

    # Map chest pain type to numeric values
    cp_mapping = {"Type 1": 1, "Type 2": 2, "Type 3": 3, "Type 4": 4}
    cp_numeric = cp_mapping[cp]

    # Predict button
    if st.button("Predict", key="predict"):
        input_data = np.array([[thalach, age, cp_numeric]])
        prediction = model.predict(input_data)
        
        # Display result
        if prediction[0] == 1:
            st.markdown("<div class='result-danger'>Heart Disease Detected. Please consult a doctor.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-success'>No Heart Disease Detected. Keep up the healthy lifestyle!</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
