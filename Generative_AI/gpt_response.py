import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Get your OpenAI API key from environment or paste directly here (for testing only)
api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

predicted_disease = "Diabetes"

prompt = f"""
Give a short and clear medical summary for the disease: {predicted_disease}.
Include:
1. Description
2. Treatment
3. Prevention
"""

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

reply = response.choices[0].message.content

# Split the response
lines = reply.split('\n')
description = treatment = prevention = ""
for line in lines:
    if line.lower().startswith("1. description"):
        description = line.split(':', 1)[-1].strip()
    elif line.lower().startswith("2. treatment"):
        treatment = line.split(':', 1)[-1].strip()
    elif line.lower().startswith("3. prevention"):
        prevention = line.split(':', 1)[-1].strip()

# Formal Template
formal_response = f"""
**Diagnosis:** {predicted_disease}

**Description:**
{description}

**Treatment:**
{treatment}

**Prevention:**
{prevention}
"""

# Friendly Template
friendly_response = f"""
Looks like you're feeling some symptoms. Based on what you entered, you might have **{predicted_disease}**!

Here's what it means:
{description}

To feel better:
{treatment}

Stay safe!
{prevention}
"""

print("\n===== FORMAL TEMPLATE =====\n")
print(formal_response)

print("\n===== FRIENDLY TEMPLATE =====\n")
print(friendly_response)
