MODEL_STATS = {
    "tiny.en": "tiny (0.80 acc | 1GB VRAM | 10x Speed)",
    "base.en": "base (0.85 acc | 1.2GB VRAM | 7x Speed)",
    "small.en": "small (0.90 acc | 2GB VRAM | 4x Speed)",
    "medium.en": "medium (0.93 acc | 3.5GB VRAM | 2x Speed)",
    "distil-large-v3": "large (0.95 acc | 4GB VRAM | 6x Speed)",
    "large-v3-turbo": "turbo (0.96 acc | 4GB VRAM | 8x Speed)",
}

HIDE_ST_STYLE = """
<style>
[data-testid="stToolbar"] {visibility: hidden !important;}
[data-testid="stHeader"] {visibility: hidden !important;}
footer {visibility: hidden !important;}
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a,
.stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a,
[data-testid="stHeaderActionElements"] {display: none !important;}
</style>
"""
