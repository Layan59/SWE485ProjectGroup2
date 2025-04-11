import pandas as pd
import joblib
from gpt_response import get_gpt_response

# Load the saved model
svm_model = joblib.load('svm_model.pkl')

# Load your training data (just for getting the symptoms list)
X_train = pd.read_csv("X_train.csv")

# Example user input
user_symptoms = ["headache", "fatigue", "high_fever", "muscle_pain"]

# Encode user symptoms same as training data
all_symptoms = X_train.columns.tolist()
input_data = [1 if symptom in user_symptoms else 0 for symptom in all_symptoms]
input_df = pd.DataFrame([input_data], columns=all_symptoms)

# Predict Disease
predicted_disease = svm_model.predict(input_df)[0]

print(f"Predicted Disease: {predicted_disease}")

# Generate GPT Response
response = get_gpt_response(predicted_disease)
print(response)
