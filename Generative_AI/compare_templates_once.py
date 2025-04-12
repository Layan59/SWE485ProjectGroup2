import os
import openai
from dotenv import load_dotenv
import joblib
import pandas as pd

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# Load the trained model and symptoms list
svm_model = joblib.load('svm_model.pkl')
X_train = pd.read_csv('X_train.csv')
all_symptoms = X_train.columns.tolist()

# Function to call GPT once and extract the core info
def get_base_gpt_response(predicted_disease):
    prompt = f"""
    Give a short and clear medical summary for the disease: {predicted_disease}.
    Include:
    1. Description
    2. Treatment
    3. Prevention
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    reply = response.choices[0].message.content
    lines = reply.split('\n')
    description = treatment = prevention = ""
    for line in lines:
        if line.lower().startswith("1. description"):
            description = line.split(':', 1)[-1].strip()
        elif line.lower().startswith("2. treatment"):
            treatment = line.split(':', 1)[-1].strip()
        elif line.lower().startswith("3. prevention"):
            prevention = line.split(':', 1)[-1].strip()

    return description, treatment, prevention

# Function to apply templates on the same GPT result
def compare_templates(predicted_disease, description, treatment, prevention):
    formal = f"""
    **Diagnosis:** {predicted_disease}

    **Description:**
    {description}

    **Treatment:**
    {treatment}

    **Prevention:**
    {prevention}
    """

    friendly = f"""
    Looks like you're feeling some symptoms. Based on what you entered, you might have **{predicted_disease}**!

    Here's what it means:
    {description}

    To feel better:
    {treatment}

    Stay safe!  
    {prevention}
    """

    return formal.strip(), friendly.strip()

# Example user input
selected_symptoms = ["fever", "fatigue", "cough"]

# Convert symptoms to model input
input_data = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
input_df = pd.DataFrame([input_data], columns=all_symptoms)

# Predict disease
predicted_disease = svm_model.predict(input_df)[0]

# Get shared GPT content
description, treatment, prevention = get_base_gpt_response(predicted_disease)

# Format both templates
formal_output, friendly_output = compare_templates(predicted_disease, description, treatment, prevention)

# Print results
print("===== FORMAL TEMPLATE =====\n")
print(formal_output)
print("\n===== FRIENDLY TEMPLATE =====\n")
print(friendly_output)


# Ask GPT to generate a comparison table between the two templates
table_prompt = f"""
Compare the following two medical summaries for the disease: {predicted_disease}.
One is written in a formal style, and the other is in a friendly style.

Please create a comparison table that covers:
- Tone
- Readability
- Emotion
- Audience Suitability

Use markdown table format only.

Formal Template:
{formal_output}

Friendly Template:
{friendly_output}
"""

table_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": table_prompt}],
    temperature=0.7
)

# Print the GPT-generated comparison table
print("\n===== GPT-GENERATED COMPARISON TABLE =====\n")
print(table_response.choices[0].message.content)
