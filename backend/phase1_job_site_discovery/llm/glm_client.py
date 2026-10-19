import os
from openai import OpenAI
from config import GLM_MODEL


def ask_glm(prompt: str, model: str = GLM_MODEL) -> str:
    """Call Zhipu AI GLM API."""
    client = OpenAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096
    )
    return response.choices[0].message.content
