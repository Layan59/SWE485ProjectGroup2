import streamlit as st
import pandas as pd
import joblib
from gpt_response import get_gpt_response
from predict_disease import predict_disease  # Import the predict_disease function



# Load model and training columns
random_forest_model = joblib.load('random_forest_model.pkl')
X_train = pd.read_csv("X_train.csv")
all_symptoms = X_train.columns.tolist()

st.set_page_config(page_title="Symptom Checker with AI", layout="centered")

st.title("🩺 Symptom Checker")
st.write("Select your symptoms and get a prediction with medical advice.")

# Symptom selector
selected_symptoms = st.multiselect("Select your symptoms:", options=all_symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Call the predict_disease function to get the disease prediction and GPT response
        predicted_disease, gpt_response = predict_disease(selected_symptoms)
        
        # Display predicted disease
        st.success(f"🧬 Predicted Disease: **{predicted_disease}**")

        # Display GPT response
        st.markdown("### 💡 GPT Medical Advice")
        st.markdown(gpt_response)


