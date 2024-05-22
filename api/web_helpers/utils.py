import random
from scipy import stats


def random_color() -> str:
    """
    Generates a random, visually appealing color in RGBA format for Plotly graphs.
    Colors are chosen to have high contrast and visibility against a white background.
    """
    colors: list[str] = [
        "rgba(214, 39, 40, 0.8)",  # Bright Red
        "rgba(31, 119, 180, 0.8)",  # Vivid Blue
        "rgba(44, 160, 44, 0.8)",  # Vivid Green
        "rgba(255, 127, 14, 0.8)",  # Bright Orange
        "rgba(148, 103, 189, 0.8)",  # Vivid Purple
        "rgba(140, 86, 75, 0.8)",  # Strong Brown
        "rgba(227, 119, 194, 0.8)",  # Pinkish
        "rgba(127, 127, 127, 0.8)",  # Gray
        "rgba(188, 189, 34, 0.8)",  # Olive Green
        "rgba(23, 190, 207, 0.8)",  # Bright Cyan
        "rgba(174, 199, 232, 0.8)",  # Soft Blue
        "rgba(255, 152, 150, 0.8)",  # Soft Red
        "rgba(197, 176, 213, 0.8)",  # Soft Purple
        "rgba(196, 156, 148, 0.8)",  # Soft Brown
        "rgba(247, 182, 210, 0.8)",  # Soft Pink
        "rgba(199, 199, 199, 0.8)",  # Light Gray
        "rgba(219, 219, 141, 0.8)",  # Light Olive
        "rgba(158, 218, 229, 0.8)",  # Light Cyan
        "rgba(255, 187, 120, 0.8)",  # Peach
        "rgba(152, 223, 138, 0.8)",  # Pale Green
        "rgba(255, 152, 150, 0.8)",  # Pale Red
        "rgba(23, 190, 207, 0.8)",  # Teal
        "rgba(158, 218, 229, 0.8)",  # Pale Cyan
        "rgba(197, 176, 213, 0.8)",  # Pale Purple
    ]

    return random.choice(colors)


def kde_scipy(x: list[float], x_grid: list[int], bandwidth: float = 0.2, **kwargs) -> list[float]:
    """Compute the kernel density estimate on a grid of x values."""
    kde = stats.gaussian_kde(x, bw_method=bandwidth, **kwargs)
    return kde.evaluate(x_grid)
