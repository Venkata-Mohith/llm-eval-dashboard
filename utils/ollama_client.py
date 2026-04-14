import ollama
import time

def get_available_models():
    """Fetch all models currently pulled in Ollama."""
    try:
        response = ollama.list()
        return [m.model for m in response.models]
    except Exception as e:
        return []

def query_model(model_name: str, prompt: str, system_prompt: str = None, timeout: int = 60) -> str:
    """
    Send a prompt to a local Ollama model and return the response text.
    Returns empty string on failure — never crashes the evaluation loop.
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        start = time.time()
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0.1}  # Low temp for consistent eval results
        )
        elapsed = round(time.time() - start, 2)
        text = response.message.content.strip()
        return text, elapsed
    except Exception as e:
        return "", 0.0