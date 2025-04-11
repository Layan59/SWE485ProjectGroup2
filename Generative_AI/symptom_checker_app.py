import streamlit as st
import pandas as pd
import joblib
from gpt_response import get_gpt_response

# Load model and training columns
svm_model = joblib.load('svm_model.pkl')
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
        # Prepare input data
        input_data = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
        input_df = pd.DataFrame([input_data], columns=all_symptoms)

        # Predict
        predicted_disease = svm_model.predict(input_df)[0]
        st.success(f"🧬 Predicted Disease: **{predicted_disease}**")

        # Get GPT advice
        gpt_response = get_gpt_response(predicted_disease, template_type="friendly")
        st.markdown("### 💡 GPT Medical Advice")
        st.markdown(gpt_response)
