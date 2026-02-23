import time
import streamlit as st
from pathlib import Path

from utils.diff_utils import generate_html_diff, build_paragraph_diffs, apply_rejection


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
        )
    with col2:
        view_mode = st.selectbox(
            "View Mode",
            [
                "Split View",
                "Diff View",
                "Interactive Review",
                "Raw Only",
                "Edited Only",
            ],
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
            st.caption(
                "Red (strikethrough) = removed · Green (bold) = added · Click ✕ to reject a change"
            )

            if "pending_rejection" not in st.session_state:
                st.session_state.pending_rejection = None

            if st.session_state.pending_rejection is not None:
                chg = st.session_state.pending_rejection
                current_text = proc_file.read_text(encoding="utf-8")
                updated_text = apply_rejection(current_text, chg)
                proc_file.write_text(updated_text, encoding="utf-8")
                st.session_state.pending_rejection = None
                st.rerun()

            para_diffs = build_paragraph_diffs(raw_text, proc_text)

            all_segments = []
            for p_idx, para in enumerate(para_diffs):
                for seg in para["segments"]:
                    all_segments.append((p_idx, seg))

            prev_p_idx = None
            for seg_idx, (p_idx, seg) in enumerate(all_segments):
                if prev_p_idx is not None and p_idx != prev_p_idx:
                    st.markdown(
                        '<hr style="border: none; border-top: 1px solid #333; margin: 6px 0;">',
                        unsafe_allow_html=True,
                    )
                prev_p_idx = p_idx

                chg = seg["change"]
                if chg is not None:
                    left_col, right_col = st.columns([3, 2])
                    with left_col:
                        st.markdown(
                            f'<div style="padding: 0.5rem 0.75rem; background-color: #1E1E1E; color: #E0E0E0; '
                            f'border-radius: 5px; line-height: 1.6; margin-bottom: 2px;">'
                            f'{seg["html"]}</div>',
                            unsafe_allow_html=True,
                        )
                    with right_col:
                        c_text, c_btn = st.columns([5, 1])
                        with c_text:
                            if chg["tag"] == "delete":
                                st.markdown(
                                    f'<div style="padding: 4px 8px; background-color: #3a1515; border-radius: 4px; '
                                    f'font-size: 0.85em; color: #f0a0a0; margin-bottom: 2px;">'
                                    f'<strong>Deleted:</strong> {chg["old"]}</div>',
                                    unsafe_allow_html=True,
                                )
                            elif chg["tag"] == "insert":
                                st.markdown(
                                    f'<div style="padding: 4px 8px; background-color: #152a15; border-radius: 4px; '
                                    f'font-size: 0.85em; color: #a0f0a0; margin-bottom: 2px;">'
                                    f'<strong>Added:</strong> {chg["new"]}</div>',
                                    unsafe_allow_html=True,
                                )
                            elif chg["tag"] == "replace":
                                st.markdown(
                                    f'<div style="padding: 4px 8px; background-color: #2a2a15; border-radius: 4px; '
                                    f'font-size: 0.85em; color: #f0e0a0; margin-bottom: 2px;">'
                                    f"<strong>Changed:</strong> "
                                    f'<span style="text-decoration: line-through; color: #f0a0a0;">{chg["old"]}</span>'
                                    f' → <span style="color: #a0f0a0;">{chg["new"]}</span></div>',
                                    unsafe_allow_html=True,
                                )
                        with c_btn:
                            st.button(
                                "✕",
                                key=f"reject_{selected_base}_{chg['id']}",
                                type="secondary",
                                help="Reject this change",
                                on_click=lambda c=chg: st.session_state.__setitem__(
                                    "pending_rejection", c
                                ),
                            )
                else:
                    st.markdown(
                        f'<div style="padding: 0.5rem 0.75rem; background-color: #1E1E1E; color: #888; '
                        f'border-radius: 5px; line-height: 1.6; margin-bottom: 2px;">'
                        f'{seg["html"]}</div>',
                        unsafe_allow_html=True,
                    )

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

    if view_mode == "Interactive Review":
        if not raw_text or not proc_text:
            st.warning(
                "You need both a Raw and an Edited version to use Interactive Review."
            )
        else:
            st.info(
                "Review changes block-by-block. Edit the final text if needed, then click 'Approve'."
            )

            raw_paragraphs = [p for p in raw_text.split("\n\n") if p.strip()]
            proc_paragraphs = [p for p in proc_text.split("\n\n") if p.strip()]

            if (
                "review_data" not in st.session_state
                or st.session_state.get("review_base") != selected_base
            ):
                st.session_state.review_base = selected_base
                st.session_state.review_data = {}
                for idx, txt in enumerate(proc_paragraphs):
                    st.session_state.review_data[idx] = txt

            max_len = max(len(raw_paragraphs), len(proc_paragraphs))

            with st.container(height=600):
                for i in range(max_len):
                    r_p = raw_paragraphs[i] if i < len(raw_paragraphs) else ""
                    p_p = proc_paragraphs[i] if i < len(proc_paragraphs) else ""

                    st.markdown(f"#### Segment {i+1}")

                    r_col, diff_col, final_col = st.columns([1, 1, 1.5])

                    with r_col:
                        st.caption("Raw Input")
                        st.text_area(
                            f"raw_{i}",
                            r_p,
                            height=150,
                            disabled=True,
                            label_visibility="collapsed",
                        )

                    with diff_col:
                        st.caption("Changes Detected")
                        diff_html = generate_html_diff(r_p, p_p)
                        st.markdown(
                            f'<div style="background-color:#0e1117; padding:10px; border-radius:5px; height:150px; overflow-y:auto; font-size:14px; line-height:1.5; border:1px solid #303030;">{diff_html}</div>',
                            unsafe_allow_html=True,
                        )

                    with final_col:
                        st.caption("Final Output (Editable)")
                        val = st.text_area(
                            f"final_{i}",
                            value=st.session_state.review_data.get(i, p_p),
                            height=150,
                            key=f"edit_area_{i}",
                            label_visibility="collapsed",
                        )
                        st.session_state.review_data[i] = val

                    st.divider()

            st.markdown("### Final Actions")
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("💾 Save Verified Transcript", type="primary"):
                    final_segments = []
                    for i in range(max_len):
                        final_segments.append(st.session_state.review_data.get(i, ""))

                    full_text = "\n\n".join(final_segments)
                    save_path = processed_dir / f"{selected_base}_VERIFIED.txt"
                    save_path.write_text(full_text, encoding="utf-8")
                    st.success(f"Saved verified transcript to: {save_path.name}")
                    time.sleep(1)
                    st.rerun()
