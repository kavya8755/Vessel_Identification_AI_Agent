from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

client = Groq(api_key=GROQ_API_KEY)

def ask_llm(question,data):

    prompt = f"""
You are a maritime analyst.

User question:
{question}

Database results:
{data}

Answer clearly.
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"user","content":prompt}
        ]
    )

    return completion.choices[0].message.content