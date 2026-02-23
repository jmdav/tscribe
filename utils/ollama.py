import logging
import streamlit as st
import requests


@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models():
    """Fetches installed models from the local Ollama instance. Cached for 30s."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except (requests.ConnectionError, requests.Timeout):
        return []
    except Exception as e:
        logging.warning(f"Failed to fetch Ollama models: {e}")
        return []
