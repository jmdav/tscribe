import streamlit as st
from faster_whisper import WhisperModel


@st.cache_resource(show_spinner=True)
def load_model(model_size: str, device: str):
    try:
        if device == "cuda":
            compute_type = "float16"
        elif device == "mps":
            # MPS doesn't support int8; use float16
            compute_type = "float16"
        else:
            compute_type = "int8"
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        st.error(f"Failed to load model '{model_size}': {e}")
        raise
