import os
from openai import OpenAI
from config import GPT_MODEL


def ask_gpt(prompt: str, model: str = GPT_MODEL) -> str:
    """Call OpenAI GPT API."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=4096
    )
    return response.choices[0].message.content
