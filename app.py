import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model_new.pkl")

@st.cache_data
def load_data():
    
    return pd.read_csv("heart_disease_data_finalized.csv")

model = load_model()
df = load_data()

# Streamlit UI
st.title("❤️ Heart Disease Risk Prediction using Machine Learning")
st.markdown("### Capstone Project")

st.info("""
This application predicts whether a person is at risk of heart disease using
Machine Learning based on important health parameters.
""")

# Create tabs to fulfill both the prediction and dashboard requirements
tab1, tab2 = st.tabs(["❤️ Predict Heart Disease", "📊 Data Analysis"])

with tab1:
    st.header("Prediction Tool")
    st.write("Enter the patient's details to predict heart disease severity.")

    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=200)
    thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    

    # Convert categorical inputs to numerical values
    sex_num = 1 if sex == "Male" else 0
    cp_num = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(cp)
    

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame(
    [[age, sex_num, cp_num, trestbps, chol, thalch]],
    columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "thalch"
    ]
    )
    input_data = input_data.apply(pd.to_numeric, errors='coerce')
    input_data = input_data.fillna(0)
   

    
    if st.button("Predict Risk"):
        prediction = model.predict(input_data)
        if prediction[0] == 0:
             st.success("✅ Low Risk of Heart Disease")
             st.info(""" ### Health Tips
                        - Maintain a balanced diet
                        - Exercise regularly
                        - Get enough sleep
                        - Monitor your blood pressure""")
        else:
             st.error("⚠️ High Risk of Heart Disease")
             st.warning("""
                            ### Recommendation
                            - Consult a cardiologist
                            - Monitor blood pressure regularly
                            - Reduce cholesterol intake
                            - Exercise as advised by your doctor
                            """)

with tab2:
    st.header("Interactive Data Dashboard")
    st.write("Explore the key findings and trends from the heart disease dataset.")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution Histogram
        fig_age = px.histogram(df, x="age", color="num", barmode="group",
                               title="Patient Age Distribution",
                               labels={"num": "Severity (0=No Disease)", "age": "Age"})
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Chest Pain Type Bar Chart
        fig_cp = px.histogram(df, x="cp", color="num", barmode="group",
                              title="Chest Pain Analysis",
                              labels={"cp": "Chest Pain Type", "num": "Severity"})
        st.plotly_chart(fig_cp, use_container_width=True)

    with col2:
        # Cholesterol vs Age Scatter Plot
        fig_chol = px.scatter(df, x="age", y="chol", color="num", 
                              title=" Age vs Cholesterol Levels  ",
                              labels={"chol": "Cholesterol (mg/dl)", "age": "Age", "num": "Severity"})
        st.plotly_chart(fig_chol, use_container_width=True)
        
        # Max Heart Rate vs Age Scatter Plot
        fig_thalach = px.scatter(df, x="age", y="thalch", color="num",
                                 title="Age vs Maximum Heart Rate",
                                 labels={"thalch": "Max Heart Rate", "age": "Age", "num": "Severity"})
        st.plotly_chart(fig_thalach, use_container_width=True)