"""Block-character rendering of the Ephemeris logo, for the CLI splash.

Precomputed (not regenerated at runtime) from the vector art in
the project's logo -- an astronomical-instrument motif
(graduated scale ring, tilted elliptical orbit, position markers, crosshair)
rasterized onto a 36x36 grid, with a wider stroke threshold than a literal
1:1 trace for a smaller, bolder mark, and packed two rows per output line
with Unicode half-block characters (▀ ▄ █) for double vertical resolution.
"""

LOGO_LINES: list[str] = [
    "         ▄▄███▀▀▀██▀▀▀███▄▄         ",
    "      ▄████      ██      ████▄      ",
    "    ▄█▀▀ ▀▀              ▀▀ ▀▀█▄    ",
    "  ▄██                          ██▄  ",
    " ▄██▄                          ▄██▄ ",
    "▄█▀▀▀          ▄▄▄█████████▄▄  ▀▀▀█▄",
    "██         ▄▄██▀▀▀ ▄██     ▀██    ██",
    "█       ▄██▀▀    ▄▄██        ██    █",
    "█▄▄▄   ██▀     ▄▄███▄       ██  ▄▄▄█",
    "█▀▀▀  ██       ▀▀██▀▀     ▄██   ▀▀▀█",
    "█    ██          ▀▀    ▄▄██▀       █",
    "██    ██▄         ▄▄▄██▀▀         ██",
    "▀█▄▄▄  ▀▀█████████▀▀▀          ▄▄▄█▀",
    " ▀██▀                          ▀██▀ ",
    "  ▀██                          ██▀  ",
    "    ▀█▄▄ ▄▄              ▄▄ ▄▄█▀    ",
    "      ▀████      ██      ████▀      ",
    "         ▀▀███▄▄▄██▄▄▄███▀▀         ",
]
