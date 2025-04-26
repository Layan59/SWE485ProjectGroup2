import pandas as pd
import joblib
from gpt_response import get_gpt_response

# Load the random forest model
random_forest_model = joblib.load('random_forest_model.pkl')

# Load training data (just for getting the symptoms list)
X_train = pd.read_csv("X_train.csv")
all_symptoms = X_train.columns.tolist()

def predict_disease(selected_symptoms):
    # Encode user symptoms same as training data
    input_data = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
    input_df = pd.DataFrame([input_data], columns=all_symptoms)

    # Predict Disease using Random Forest
    predicted_disease = random_forest_model.predict(input_df)[0]

    # Generate GPT Response
    gpt_response = get_gpt_response(predicted_disease, template_type="friendly")
    
    return predicted_disease, gpt_response

if __name__ == "__main__":
    print("Please enter symptoms separated by commas (e.g., headache,dizziness,nausea):")
    user_input = input()
    selected_symptoms = [symptom.strip() for symptom in user_input.split(",")]

    predicted_disease, gpt_response = predict_disease(selected_symptoms)

    print("\nPredicted Disease:", predicted_disease)
    print("AI Advice:", gpt_response)
