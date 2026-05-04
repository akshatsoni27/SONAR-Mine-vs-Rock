# 🌊 SONAR Rock vs Mine Prediction 🛑

A Machine Learning powered application to classify sonar returns as either bouncing off a metal cylinder (Mine) or a roughly cylindrical rock.

## 🚀 Overview
This project uses a **Logistic Regression** model trained on sonar data to predict whether an object is a Rock or a Mine based on 60 frequency band energy inputs. It features an interactive web interface built with **Streamlit** for real-time predictions.

## 🛠️ Technologies Used
* **Python**: Core programming language.
* **Streamlit**: Web framework for building the interactive UI.
* **Scikit-Learn**: Machine learning library used for building and evaluating the Logistic Regression model.
* **Pandas & NumPy**: Data manipulation and numerical operations.

## 📁 Project Structure
* `app.py`: The main Streamlit web application script. It handles data loading, model training, and the web interface.
* `Rock_vs_Mine_Prediction.ipynb`: Jupyter notebook containing the exploratory data analysis (EDA) and initial model training.
* `sonar data.csv`: The dataset containing 208 samples, each with 60 features and 1 label column (`R` for Rock, `M` for Mine).
* `requirements.txt`: List of Python dependencies required to run the project.

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akshatsoni27/SONAR-Mine-vs-Rock.git
   cd SONAR-Mine-vs-Rock
   ```

2. **Create and activate a virtual environment (Optional but recommended):**
   ```bash
   python -m venv .venv
   
   # Windows:
   .\.venv\Scripts\activate
   
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage
Start the Streamlit application by running the following command in your terminal:

```bash
streamlit run app.py
```

This will launch a local web server and automatically open the application in your default web browser (typically at `http://localhost:8501`).

### How to Predict:
1. Copy a comma-separated list of exactly 60 numerical sonar features.
2. Paste it into the application's input field.
3. Click the **Predict Object 🎯** button to see if the sonar detected a **Rock** or a **Mine**.

---
*Created by [akshatsoni27](https://github.com/akshatsoni27)*
