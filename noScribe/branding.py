"""Colour palette of Stichting Mijn Aansluiting, shared by the GUI and PDF export.

Values are sampled from the foundation's logo. Keep all brand colours here so
the GUI theme and the PDF export stay consistent.
"""

BLUE = "#1D4F9C"        # main colour: word mark and middle figure
BLUE_DARK = "#163D79"   # hover / pressed state of blue elements
BLUE_DEEP = "#0F2A52"   # dark surfaces (header) in the dark appearance
MAGENTA = "#BC2C80"     # left figure, used for recording
MAGENTA_DARK = "#951F64"
CYAN = "#1EB5D8"        # top dot, used for accents and progress
GREEN = "#149B6E"       # right figure, used for "done" and the start button
GREEN_DARK = "#0F7853"
GREY_TEXT = "#6B7785"   # secondary text such as timestamps

# Order of the figures in the logo, left to right, used for decorative stripes.
LOGO_STRIPE = (MAGENTA, CYAN, BLUE, GREEN)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def apply_ctk_theme(ctk) -> None:
    """Replace the blue accents of customtkinter's 'blue' theme by brand colours.

    Call after ``ctk.set_default_color_theme('blue')`` and before any widget is
    created. Each value is a [light, dark] pair.
    """
    theme = ctk.ThemeManager.theme
    main = [BLUE, BLUE]
    hover = [BLUE_DARK, BLUE_DARK]
    for widget in ("CTkButton", "CTkCheckBox", "CTkRadioButton"):
        theme[widget]["fg_color"] = main
        theme[widget]["hover_color"] = hover
    theme["CTkSwitch"]["progress_color"] = main
    theme["CTkProgressBar"]["progress_color"] = [CYAN, CYAN]
    theme["CTkSlider"]["button_color"] = main
    theme["CTkSlider"]["button_hover_color"] = hover
    theme["CTkOptionMenu"]["fg_color"] = main
    theme["CTkOptionMenu"]["button_color"] = hover
    theme["CTkOptionMenu"]["button_hover_color"] = [BLUE_DEEP, BLUE_DEEP]
    theme["CTkSegmentedButton"]["selected_color"] = main
    theme["CTkSegmentedButton"]["selected_hover_color"] = hover


def header_graphic_on(image, background: str = BLUE_DEEP):
    """Turn the grey-on-grey header graphic into light glyphs on ``background``.

    The original graphic is light grey text on a #2b2b2b background; the
    brightness above that background becomes the opacity of white glyphs.
    """
    from PIL import Image

    grey = image.convert("L")
    alpha = grey.point(lambda v: max(0, min(255, int((v - 43) * 1.6))))
    out = Image.new("RGB", image.size, hex_to_rgb(background))
    glyphs = Image.new("RGB", image.size, (210, 225, 240))
    out.paste(glyphs, (0, 0), alpha)
    return out
