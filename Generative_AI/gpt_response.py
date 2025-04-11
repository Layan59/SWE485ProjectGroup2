import os
import openai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

def get_gpt_response(predicted_disease, template_type="formal"):
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

    To avoid it in the future:
    {prevention}
    """

    if template_type == "formal":
        return formal_response.strip()
    else:
        return friendly_response.strip()
