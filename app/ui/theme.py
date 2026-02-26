"""
Theme — Dark security-first Gradio theme for PrivateSearch.
Communicates privacy and trust through visual design.
"""

import gradio as gr


def create_theme() -> gr.Theme:
    """Create a dark, professional theme that communicates security."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e8f5e9",
            c100="#c8e6c9",
            c200="#a5d6a7",
            c300="#81c784",
            c400="#66bb6a",
            c500="#4caf50",   # Primary green — security color
            c600="#43a047",
            c700="#388e3c",
            c800="#2e7d32",
            c900="#1b5e20",
            c950="#0d3311",
        ),
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        # ─── Background ────
        body_background_fill="#0f1117",
        body_background_fill_dark="#0f1117",
        background_fill_primary="#161922",
        background_fill_primary_dark="#161922",
        background_fill_secondary="#1c2030",
        background_fill_secondary_dark="#1c2030",
        # ─── Borders ────
        border_color_primary="#2a2f3a",
        border_color_primary_dark="#2a2f3a",
        border_color_accent="#4caf50",
        border_color_accent_dark="#4caf50",
        # ─── Text ────
        body_text_color="#e0e0e0",
        body_text_color_dark="#e0e0e0",
        body_text_color_subdued="#8891a5",
        body_text_color_subdued_dark="#8891a5",
        # ─── Buttons ────
        button_primary_background_fill="#2e7d32",
        button_primary_background_fill_dark="#2e7d32",
        button_primary_background_fill_hover="#388e3c",
        button_primary_background_fill_hover_dark="#388e3c",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#1c2030",
        button_secondary_background_fill_dark="#1c2030",
        button_secondary_text_color="#e0e0e0",
        button_secondary_text_color_dark="#e0e0e0",
        # ─── Inputs ────
        input_background_fill="#1c2030",
        input_background_fill_dark="#1c2030",
        input_border_color="#2a2f3a",
        input_border_color_dark="#2a2f3a",
        input_border_color_focus="#4caf50",
        input_border_color_focus_dark="#4caf50",
        # ─── Blocks ────
        block_background_fill="#161922",
        block_background_fill_dark="#161922",
        block_border_color="#2a2f3a",
        block_border_color_dark="#2a2f3a",
        block_label_text_color="#8891a5",
        block_label_text_color_dark="#8891a5",
        block_title_text_color="#e0e0e0",
        block_title_text_color_dark="#e0e0e0",
        # ─── Shadows ────
        shadow_drop="0 2px 8px rgba(0,0,0,0.3)",
        shadow_drop_lg="0 4px 16px rgba(0,0,0,0.4)",
    )


# ─── Custom CSS for extra styling ─────────────────────────────────────────

CUSTOM_CSS = """
/* Security header banner */
.security-banner {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
    border: 1px solid #4caf50;
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 16px;
    text-align: center;
    font-size: 14px;
    color: #e8f5e9;
    letter-spacing: 0.5px;
}

.security-banner .lock-icon {
    font-size: 18px;
    margin-right: 8px;
}

/* App title */
.app-title {
    font-size: 28px;
    font-weight: 700;
    color: #e0e0e0;
    margin: 0;
    padding: 0;
}

.app-subtitle {
    font-size: 14px;
    color: #8891a5;
    margin-top: 4px;
}

/* Stats panel */
.stats-panel {
    background: #1c2030;
    border: 1px solid #2a2f3a;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #8891a5;
}

.stats-panel .stat-value {
    color: #e0e0e0;
    font-weight: 600;
}

/* Status indicator */
.status-ready {
    color: #4caf50;
    font-weight: 600;
}

.status-busy {
    color: #ff9800;
    font-weight: 600;
}

.status-error {
    color: #f44336;
    font-weight: 600;
}

/* Sources accordion */
.source-item {
    background: #1c2030;
    border: 1px solid #2a2f3a;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
}

.source-filename {
    color: #4caf50;
    font-weight: 600;
}

.source-score {
    color: #8891a5;
    float: right;
}

.source-preview {
    color: #8891a5;
    margin-top: 6px;
    font-size: 12px;
    line-height: 1.4;
}

/* Footer */
.privacy-footer {
    text-align: center;
    color: #5a6275;
    font-size: 12px;
    padding: 16px;
    border-top: 1px solid #2a2f3a;
    margin-top: 16px;
}

/* Change detection panel */
.changes-panel {
    background: #1c2030;
    border: 1px solid #ff9800;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}

.changes-new { color: #4caf50; }
.changes-modified { color: #ff9800; }
.changes-deleted { color: #f44336; }

/* Progress styling */
.progress-container {
    background: #1c2030;
    border-radius: 8px;
    padding: 16px;
    margin: 10px 0;
}

/* Hide Gradio footer */
footer { display: none !important; }

/* Chatbot message styling */
.message-wrap {
    font-size: 14px !important;
}
"""
