import logging
import streamlit as st
from pathlib import Path

from config import HIDE_ST_STYLE
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

# --- APPLICATION-WIDE CONFIG ---
with st.expander("Application Settings", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Input Directory**")
        input_dir_str = st.text_input(
            "Input (media files)",
            value="in",
            label_visibility="collapsed",
            help="Where media files (MP4, MP3, WAV, etc.) are located for transcription.",
        )

    with col2:
        st.markdown("**Output Directory**")
        output_dir_str = st.text_input(
            "Output (raw transcripts)",
            value="out",
            label_visibility="collapsed",
            help="Where raw transcripts from speech-to-text are saved.",
        )

    with col3:
        st.markdown("**Processed Directory**")
        processed_dir_str = st.text_input(
            "Processed (edited transcripts)",
            value="processed",
            label_visibility="collapsed",
            help="Where AI-edited transcripts are saved after grammar/formatting improvements.",
        )

input_dir = Path(input_dir_str).expanduser().resolve()
output_dir = Path(output_dir_str).expanduser().resolve()
processed_dir = Path(processed_dir_str).expanduser().resolve()

# Validate and create all directories
errors = []
try:
    validate_directory(input_dir, create=True)
    validate_directory(output_dir, create=True)
    validate_directory(processed_dir, create=True)
except Exception as e:
    errors.append(f"Error with directories: {e}")

for error in errors:
    st.error(error)

# --- TABS ---
tab_transcribe, tab_process, tab_edit = st.tabs(["Transcribe", "Process", "Edit"])

with tab_transcribe:
    render_transcribe_tab(input_dir, output_dir, processed_dir)

with tab_process:
    render_edit_tab(output_dir, processed_dir)

with tab_edit:
    render_review_tab(output_dir, processed_dir)
