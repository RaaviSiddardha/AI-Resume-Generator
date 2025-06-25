# Configuration system

__all__ = [
    "get_st_config", "update_st_config", "llm_models", "embedding_models", "default_llm", "default_embedding", "get_llm_config", "get_embedding_config"
]

llm_models = {
    "google/flan-t5-small": {"provider": "huggingface"},
    "microsoft/DialoGPT-medium": {"provider": "huggingface"},
    "gpt-3.5-turbo": {"provider": "openai"},
    "gpt-4": {"provider": "openai"}
}
embedding_models = {
    "simple-embeddings": {"provider": "custom"},
    "all-MiniLM-L6-v2": {"provider": "huggingface"},
    "all-mpnet-base-v2": {"provider": "huggingface"},
    "openai-ada": {"provider": "openai"}
}
default_llm = "google/flan-t5-small"
default_embedding = "simple-embeddings"

_config = {
    "llm": {
        "name": default_llm,
        "provider": llm_models[default_llm]["provider"]
    },
    "embedding": {
        "name": default_embedding,
        "provider": embedding_models[default_embedding]["provider"]
    },
    "theme": "modern"
}

def get_st_config():
    """Return Streamlit or app configuration settings as a dictionary or object."""
    return _config.copy()

def update_st_config(updates: dict):
    """Update the Streamlit/app configuration with a dictionary of updates."""
    _config.update(updates)

def get_llm_config():
    """Return the current LLM config as a dict."""
    return _config["llm"]

def get_embedding_config():
    """Return the current embedding config as a dict."""
    return _config["embedding"]
