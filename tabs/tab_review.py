import html
import time
import streamlit as st
from pathlib import Path

from utils.diff_utils import generate_html_diff, build_paragraph_diffs, apply_rejection
from components.clickable_diff import clickable_diff


@st.fragment
def render_diff_view(
    selected_base: str, raw_text: str, proc_text: str, proc_file: Path
):
    """Isolated diff view that reruns only this section when rejections happen."""

    def _rerun_fragment() -> None:
        st.rerun(scope="fragment")

    # Initialize rejection history in session state
    if "rejection_history" not in st.session_state:
        st.session_state.rejection_history = {}
    if selected_base not in st.session_state.rejection_history:
        st.session_state.rejection_history[selected_base] = []

    # Track revision for forcing component refresh
    if "diff_revision" not in st.session_state:
        st.session_state.diff_revision = {}
    if selected_base not in st.session_state.diff_revision:
        st.session_state.diff_revision[selected_base] = 0

    # --- UI Rendering ---
    # Re-read proc_text to get the latest version for this render
    current_proc_text = (
        proc_file.read_text(encoding="utf-8") if proc_file.exists() else proc_text
    )

    # Undo button
    undo_col1, undo_col2 = st.columns([8, 1])
    with undo_col2:
        if st.session_state.rejection_history[selected_base]:
            undo_clicked = st.button(
                "↶ Undo",
                key=f"undo_{selected_base}",
                help="Undo the last rejection",
                type="secondary",
            )
            if undo_clicked:
                previous_text = st.session_state.rejection_history[selected_base].pop()
                proc_file.write_text(previous_text, encoding="utf-8")
                st.session_state.diff_revision[selected_base] += 1
                _rerun_fragment()

    st.caption(
        "Click changes to mark for rejection · Red = removed · Green = added (including line breaks) · Click 'Save Changes' to apply"
    )

    # Build continuous diff
    para_diffs = build_paragraph_diffs(raw_text, current_proc_text)

    # Flatten all segments (now just one "paragraph" containing everything)
    all_segments = []
    for para in para_diffs:
        if para.get("segments"):
            for seg in para["segments"]:
                all_segments.append(seg)

    # Render the clickable diff component with revision in key to force refresh
    revision = st.session_state.diff_revision[selected_base]
    save_action = clickable_diff(
        segments=all_segments,
        raw_text=raw_text,
        edited_text=current_proc_text,
        height=600,
        key=f"diff_{selected_base}_v{revision}",
    )

    # Handle save action - apply all rejected changes in batch or save direct edits
    if save_action is not None and save_action.get("action") == "save":
        edited_text = save_action.get("edited_text")
        rejected_changes = save_action.get("rejected_changes", [])

        if edited_text is not None or rejected_changes:
            # Save current state to history before applying rejections/edits
            st.session_state.rejection_history[selected_base].append(current_proc_text)

            if edited_text is not None:
                updated_text = edited_text
            else:
                updated_text = current_proc_text
                for change in rejected_changes:
                    updated_text = apply_rejection(updated_text, change)

            # Write the updated text to file
            proc_file.write_text(updated_text, encoding="utf-8")
            # Increment revision to force component refresh with new content
            st.session_state.diff_revision[selected_base] += 1
            _rerun_fragment()


def render_review_tab(output_dir: Path, processed_dir: Path):
    st.subheader("Transcript Inspector")

    all_bases = set()
    raw_map = {}
    proc_map = {}

    if output_dir.exists():
        for f in output_dir.rglob("*.txt"):
            base = f.stem
            all_bases.add(base)
            raw_map[base] = f

    if processed_dir.exists():
        for f in processed_dir.rglob("*.txt"):
            match = None
            for raw_base in raw_map:
                if f.name.startswith(raw_base):
                    if match is None or len(raw_base) > len(match):
                        match = raw_base

            if match:
                proc_map[match] = f
                all_bases.add(match)

    if not all_bases:
        st.info("No transcripts found. Transcribe some files first.")
        return

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected_base = st.selectbox(
            "Select Recording",
            sorted(list(all_bases)),
            index=0,
            format_func=lambda x: (f"[EDITED] {x}" if x in proc_map else f"[RAW] {x}"),
            help="Choose which transcript to review. [EDITED] shows AI-processed version, [RAW] shows original transcription.",
        )
    with col2:
        view_mode = st.selectbox(
            "View Mode",
            [
                "Diff View",
                "Split View",
                "Raw Only",
                "Edited Only",
            ],
            help="Split View: side-by-side comparison. Diff View: highlights changes with undo support. Raw/Edited Only: single version.",
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⟳", key="refresh_review", help="Refresh file list"):
            st.rerun()

    st.divider()

    raw_file = raw_map.get(selected_base)
    proc_file = proc_map.get(selected_base)

    raw_text = raw_file.read_text(encoding="utf-8") if raw_file else ""
    proc_text = proc_file.read_text(encoding="utf-8") if proc_file else ""

    if view_mode == "Split View":
        left, right = st.columns(2)
        with left:
            st.markdown("### Raw Transcript")
            if raw_file:
                st.caption(f"Source: {raw_file.name}")
                st.text_area("Raw", raw_text, height=600, label_visibility="collapsed")
            else:
                st.warning("No raw transcript found.")
        with right:
            st.markdown("### AI Polished")
            if proc_file:
                st.caption(f"Source: {proc_file.name}")
                st.text_area(
                    "Edited", proc_text, height=600, label_visibility="collapsed"
                )
            else:
                st.info("No AI-processed version yet. Go to the 'Edit' tab.")

    elif view_mode == "Diff View":
        if raw_text and proc_text:
            st.markdown("### Difference Highlight")
            render_diff_view(selected_base, raw_text, proc_text, proc_file)
        elif not proc_text:
            st.info("No edited version available to compare. Go to the 'Edit' tab.")
        else:
            st.warning("No raw transcript found for comparison.")

    elif view_mode == "Raw Only":
        st.markdown(f"### Raw Transcript: {selected_base}")
        st.text_area("Raw Content", raw_text, height=700, label_visibility="collapsed")

    elif view_mode == "Edited Only":
        if proc_text:
            st.markdown(f"### AI Polished: {selected_base}")
            st.text_area(
                "Edited Content", proc_text, height=700, label_visibility="collapsed"
            )
        else:
            st.warning("No edited content available.")
