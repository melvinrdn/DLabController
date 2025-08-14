"""
Utilities for DLab diagnostics:
- Custom matplotlib colormaps
- Logging format constants
"""

from matplotlib.colors import LinearSegmentedColormap

# ---------- Colormaps ----------

def white_turbo(segments: int) -> LinearSegmentedColormap:
    """
    Return a LinearSegmentedColormap starting from white,
    progressing through blue, turquoise, green, yellow, orange, and red.

    Parameters
    ----------
    segments : int
        Number of discrete color segments in the colormap.

    Returns
    -------
    LinearSegmentedColormap
        The generated colormap.
    """
    colors = [
        (1, 1, 1),     # white
        (0, 0, 0.5),   # dark blue
        (0, 0, 1),     # blue
        (0, 1, 1),     # turquoise
        (0, 1, 0),     # green
        (1, 1, 0),     # yellow
        (1, 0.5, 0),   # orange
        (1, 0, 0),     # red
        (0.5, 0, 0),   # dark red
    ]
    return LinearSegmentedColormap.from_list("white_turbo", colors, N=segments)


def black_red(segments: int) -> LinearSegmentedColormap:
    """
    Return a simple black-to-red colormap.

    Parameters
    ----------
    segments : int
        Number of discrete color segments in the colormap.

    Returns
    -------
    LinearSegmentedColormap
        The generated colormap.
    """
    colors = [
        (0, 0, 0),  # black
        (1, 0, 0),  # red
    ]
    return LinearSegmentedColormap.from_list("black_red", colors, N=segments)


# ---------- Gui Logging ----------

GUI_LOG_FORMAT = "[%(asctime)s] %(message)s"
GUI_LOG_DATE_FORMAT = "%H:%M:%S"
