import os
import openai

# ------------------ GPT Response Generator ------------------
# This script sends the predicted disease to GPT and returns a formatted response
# Make sure OPENAI_API_KEY is set in the GitHub Secrets or environment variables
# الكي بعطيه ليان تحطه في rep 
# Get the API key securely (from GitHub Secrets or .env)
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Example: predicted disease from ML model
predicted_disease = "Diabetes" # ترا ذا توقع كذا بس لان مابعد انربط المودل حقنا اللي يتوقع عشان يرسل ل gpt

# Prompt for GPT to generate description, treatment, and prevention
prompt = f"""
Give a short and clear medical summary for the disease: {predicted_disease}.
Include:
1. Description
2. Treatment
3. Prevention
"""

# Call GPT API
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

# Get GPT response
reply = response['choices'][0]['message']['content']

# Split GPT response by newlines to extract the 3 sections
lines = reply.split('\n')
description = treatment = prevention = ""
for line in lines:
    if line.lower().startswith("1. description"):
        description = line.split(':', 1)[-1].strip()
    elif line.lower().startswith("2. treatment"):
        treatment = line.split(':', 1)[-1].strip()
    elif line.lower().startswith("3. prevention"):
        prevention = line.split(':', 1)[-1].strip()

# Template 1 – Formal
formal_response = f"""
**Diagnosis:** {predicted_disease}

**Description:**
{description}

**Treatment:**
{treatment}

**Prevention:**
{prevention}
"""

# Template 2 – Friendly
friendly_response = f"""
Looks like you're feeling some symptoms. Based on what you entered, you might have **{predicted_disease}**!

Here's what it means:
{description}

To feel better:
{treatment}

Stay safe!
{prevention}
"""

# Print both response templates
print("\n===== FORMAL TEMPLATE =====\n")
print(formal_response)

print("\n===== FRIENDLY TEMPLATE =====\n")
print(friendly_response)
