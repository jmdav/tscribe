"""
Clickable Diff Component for Streamlit

A custom component that renders diff text with clickable change highlights.
Clicking on a highlighted change triggers a rejection callback.
"""

import os
import streamlit.components.v1 as components

# Determine if we're in development mode or production
_RELEASE = True  # Set to False during development

if not _RELEASE:
    _component_func = components.declare_component(
        "clickable_diff",
        url="http://localhost:3001",  # React dev server
    )
else:
    # Production: use built files
    _parent_dir = os.path.dirname(os.path.abspath(__file__))
    _build_dir = os.path.join(_parent_dir, "frontend", "build")
    _component_func = components.declare_component("clickable_diff", path=_build_dir)


def clickable_diff(
    segments: list,
    raw_text: str,
    edited_text: str,
    key: str = None,
    height: int = None,
) -> dict | None:
    """
    Render diff text with clickable change highlights.

    Parameters
    ----------
    segments : list
        List of segment dictionaries, each containing:
        - html: str - The HTML content to display
        - change: dict | None - Change data if this segment contains a change
            - id: int - Unique change identifier
            - tag: str - 'delete', 'insert', or 'replace'
            - old: str - Original text
            - new: str - New text
    raw_text : str
        The original raw text before edits
    edited_text : str
        The edited text with changes applied
    key : str, optional
        Unique key for the component instance
    height : int, optional
        Fixed height in pixels (scrollable if content exceeds)

    Returns
    -------
    dict | None
        If changes were saved, returns:
        {
            "action": "save",
            "rejected_changes": [list of change dicts]
        }
        Otherwise returns None.
    """
    component_value = _component_func(
        segments=segments,
        raw_text=raw_text,
        edited_text=edited_text,
        height=height,
        key=key,
        default=None,
    )
    return component_value
