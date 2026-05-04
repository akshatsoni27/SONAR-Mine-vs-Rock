import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Sonar Prediction UI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .rock {
        background: linear-gradient(135deg, #7f8c8d 0%, #bdc3c7 100%);
        color: #2c3e50;
    }
    .mine {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
    }
    h1 {
        color: #3498db;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .sub-text {
        text-align: center;
        color: #95a5a6;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Model Loading and Training ---
@st.cache_resource
def load_and_train_model():
    try:
        # Check if file exists to handle potential path issues
        file_path = 'sonar data.csv'
        if not os.path.exists(file_path):
            st.error(f"Data file not found at: {file_path}. Please ensure 'sonar data.csv' is in the same directory.")
            return None, 0.0
            
        sonar_data = pd.read_csv(file_path, header=None)
        
        # Separating labels and data
        X = sonar_data.drop(columns=60)
        Y = sonar_data[60]
        
        # Training and test data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
        
        # Model Training ---> LR
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        
        # Accuracy Evaluation
        X_test_prediction = model.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
        
        return model, test_data_accuracy
    except Exception as e:
        st.error(f"Error loading and training model: {e}")
        return None, 0.0

model, accuracy = load_and_train_model()

# --- Main UI ---
st.markdown("<h1>🌊 Rock vs Mine Prediction 🛑</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Machine Learning powered Sonar Data Classification</p>", unsafe_allow_html=True)

# Sidebar with Info
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1544551763-46a013bb70d5?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.markdown("### About the App")
    st.info(
        "This application uses a Logistic Regression model to classify sonar signals bouncing off "
        "either a metal cylinder (Mine) or a roughly cylindrical rock."
    )
    st.markdown(f"**Model Accuracy:** `{accuracy:.2%}`")
    st.markdown("---")
    st.markdown("### Example Data (Mine)")
    st.code("0.0530,0.0885,0.1997,0.2604,0.3225,0.2247,0.0617,0.2287,0.0950,0.0740,0.1610,0.2226,0.2703,0.3365,0.4266,0.4144,0.5655,0.6921,0.8547,0.9234,0.9171,1.0000,0.9532,0.9101,0.8337,0.7053,0.6534,0.4483,0.2460,0.2020,0.1446,0.0994,0.1510,0.2392,0.4434,0.5023,0.4441,0.4571,0.3927,0.2900,0.3408,0.4990,0.3632,0.1387,0.1800,0.1299,0.0523,0.0817,0.0469,0.0114,0.0299,0.0244,0.0199,0.0257,0.0082,0.0151,0.0171,0.0146,0.0134,0.0056")

# Input Section
st.markdown("### 📡 Enter Sonar Data Features")
st.markdown("Please input 60 comma-separated numerical values representing the sonar signal frequencies.")

input_data_str = st.text_area(
    "Sonar Features (CSV format)", 
    height=150,
    placeholder="e.g., 0.0200, 0.0371, 0.0428, ..."
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("Predict Object 🎯")

if predict_btn:
    if not model:
        st.error("Model is not loaded properly. Cannot predict.")
    elif not input_data_str.strip():
        st.warning("Please enter some data to predict.")
    else:
        try:
            # Parse input string to list of floats
            input_list = [float(x.strip()) for x in input_data_str.split(',')]
            
            if len(input_list) != 60:
                st.error(f"Expected exactly 60 features, but got {len(input_list)}. Please check your input.")
            else:
                # Convert to numpy array and reshape
                input_data_as_numpy_array = np.asarray(input_list)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                
                with st.spinner('Analyzing Sonar Data...'):
                    prediction = model.predict(input_data_reshaped)
                
                # Display Result
                if prediction[0] == 'R':
                    st.markdown(
                        "<div class='prediction-box rock'>"
                        "🪨 The object is a ROCK"
                        "</div>", 
                        unsafe_allow_html=True
                    )
                    st.balloons()
                else:
                    st.markdown(
                        "<div class='prediction-box mine'>"
                        "💣 WARNING: The object is a MINE!"
                        "</div>", 
                        unsafe_allow_html=True
                    )
        except ValueError:
            st.error("Invalid input. Please ensure all 60 values are numbers separated by commas.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
