from llm.claude_client import ask_claude
from llm.openai_client import ask_gpt
from llm.glm_client import ask_glm
from llm.ollama_client import ask_ollama

__all__ = ["ask_claude", "ask_gpt", "ask_glm", "ask_ollama", "ask_model"]


def ask_model(prompt: str, model_choice: str = "gpt") -> str:
    """Call LLM API.

    model_choice: "gpt" | "claude" | "glm" | "ollama"
    """
    if model_choice == "claude":
        return ask_claude(prompt)
    elif model_choice == "glm":
        return ask_glm(prompt)
    elif model_choice == "ollama":
        return ask_ollama(prompt)
    else:
        return ask_gpt(prompt)
