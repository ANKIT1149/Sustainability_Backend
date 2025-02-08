from dotenv import load_dotenv
import os
import openai


dotenv_path = "backend/.env"
load_dotenv(dotenv_path=dotenv_path)


OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
openai.api_key = OPENAI_API_KEY


def generate_answer_from_waste(waste_type):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a expert in waste management"},
                {"role": "user", "content": f"How should i recycle {waste_type}?"},
            ],
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("Openai error", e)
        return "Recycling instructions not available."
