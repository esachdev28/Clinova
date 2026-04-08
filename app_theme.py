"""
FinomIQ — Production-grade FinTech UI Theme.
Dark theme with neon accents and high-contrast typography.
"""

from gradio.themes import Base
from gradio.themes.utils import colors, fonts, sizes

class FinomIQTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.blue,
            secondary_hue=colors.zinc,
            neutral_hue=colors.zinc,
            font=fonts.GoogleFont("Inter"),
            font_mono=fonts.GoogleFont("JetBrains Mono"),
        )
        self.set(
            # Backgrounds
            body_background_fill="#0a0a0a",          # Deep Black
            body_background_fill_dark="#0a0a0a",
            block_background_fill="#121212",         # Charcoal Panel
            block_background_fill_dark="#121212",

            # Borders
            block_border_color="#333333",            # Subtle Border
            block_border_color_dark="#333333",
            border_color_accent="#00D4FF",           # Neon Blue

            # Text
            block_label_text_color="#888888",
            block_label_text_color_dark="#888888",
            block_title_text_color="#E0E0E0",
            block_title_text_color_dark="#E0E0E0",
            body_text_color="#E0E0E0",
            body_text_color_dark="#E0E0E0",

            # Buttons (Neon Blue & Green)
            button_primary_background_fill="#00D4FF",
            button_primary_background_fill_hover="#00A3CC",
            button_primary_text_color="#000000",
            
            button_secondary_background_fill="#1A1A1A",
            button_secondary_background_fill_dark="#1A1A1A",
            button_secondary_background_fill_hover="#2A2A2A",
            button_secondary_text_color="#E0E0E0",
            button_secondary_text_color_dark="#E0E0E0",

            # Inputs
            input_background_fill="#0F0F0F",
            input_background_fill_dark="#0F0F0F",
            input_border_color="#333333",
            input_border_color_dark="#333333",

            # Layout
            block_shadow="0 4px 15px rgba(0,0,0,0.5)",
            section_header_text_size=sizes.text_md,
        )
