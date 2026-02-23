import logging
import streamlit as st
import torch
from pathlib import Path

from config import MODEL_STATS, HIDE_ST_STYLE
from utils.file_utils import validate_directory
from tabs.tab_transcribe import render_transcribe_tab
from tabs.tab_review import render_review_tab
from tabs.tab_edit import render_edit_tab

logging.basicConfig(level=logging.WARNING)

# --- PAGE CONFIG ---
st.set_page_config(page_title="tScribe", page_icon="📎", layout="wide")
st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<br>", unsafe_allow_html=True)
st.title("tScribe")

# --- CONFIGURATION ---
with st.expander("Config", expanded=True):
    col_left, col_right = st.columns(2)

    with col_left:
        st.caption("DIRECTORIES")
        input_dir_str = st.text_input("Input", value="in")
        output_dir_str = st.text_input("Output", value="out")
        processed_dir_str = st.text_input("Processed", value="processed")

    with col_right:
        st.caption("TRANSCRIPTION")
        model_size = st.selectbox(
            "Model",
            options=list(MODEL_STATS.keys()),
            format_func=lambda x: MODEL_STATS[x],
            index=5,
            help="Select an optimized model based on your available RAM/VRAM.",
        )
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.insert(0, "cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_options.insert(0, "mps")
        dev_col, vad_col = st.columns(2)
        with dev_col:
            device = st.selectbox("Device", device_options)
        with vad_col:
            enable_vad = st.selectbox(
                "VAD Filter",
                options=[True, False],
                index=0,
                format_func=lambda x: "Enabled" if x else "Disabled",
                help="Filters out silence before transcribing.",
            )
        if enable_vad:
            vad_silence_ms = st.number_input(
                "VAD Min Silence (ms)", value=2000, step=500
            )
        else:
            vad_silence_ms = 2000

input_dir = Path(input_dir_str).expanduser().resolve()
output_dir = Path(output_dir_str).expanduser().resolve()
processed_dir = Path(processed_dir_str).expanduser().resolve()

# Validate all directories
errors = []
if not validate_directory(input_dir, create=True):
    errors.append(f"Cannot access input directory: {input_dir}")
if not validate_directory(output_dir, create=True):
    errors.append(f"Cannot access output directory: {output_dir}")
if not validate_directory(processed_dir, create=True):
    errors.append(f"Cannot access processed directory: {processed_dir}")

for error in errors:
    st.error(error)

# --- TABS ---
tab_transcribe, tab_viewer, tab_edit = st.tabs(["Transcribe", "Transcripts", "Edit"])

with tab_transcribe:
    render_transcribe_tab(
        input_dir,
        output_dir,
        processed_dir,
        model_size,
        device,
        enable_vad,
        vad_silence_ms,
    )

with tab_viewer:
    render_review_tab(output_dir, processed_dir)

with tab_edit:
    render_edit_tab(output_dir, processed_dir)
