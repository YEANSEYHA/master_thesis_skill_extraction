from anthropic import Anthropic
from config import CLAUDE_MODEL


def ask_claude(prompt: str, model: str = CLAUDE_MODEL) -> str:
    """Call Claude API."""
    client = Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text
