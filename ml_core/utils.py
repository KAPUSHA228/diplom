"""Общие утилиты для ml_core"""


def save_plotly_fig(fig, filename="plot", format="png"):
    """Сохранение Plotly фигуры в PNG или SVG."""
    if format == "png":
        fig.write_image(f"{filename}.png", scale=2)
    elif format == "svg":
        fig.write_image(f"{filename}.svg")
    return f"{filename}.{format}"
