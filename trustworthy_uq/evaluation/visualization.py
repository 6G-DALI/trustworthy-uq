from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def write_fig(fig, path_base, width=800, height=800):
    path_base = Path(path_base).with_suffix("")
    html = path_base.with_suffix(".html")
    png = path_base.with_suffix(".png")
    fig.write_html(str(html))
    try:
        fig.write_image(str(png), width=width, height=height, scale=2)
    except Exception:
        pass
    return html


def _normalize_position(position, name):
    """Normalize a 2-D position from (2,) or (1,2) to (2,)."""
    arr = np.asarray(position, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (2,):
            raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
        return arr
    if arr.ndim == 2:
        if arr.shape[1] != 2 or arr.shape[0] != 1:
            raise ValueError(f"{name} must have shape (2,) or (1,2), got {arr.shape}")
        return arr[0]
    raise ValueError(f"{name} must have shape (2,) or (1,2), got {arr.shape}")


def plot_single_sample(center, true, radius, sla_level, method_name="CQR-OneSided", output_path=None):
    center = _normalize_position(center, "center")
    true = _normalize_position(true, "true")
    radius = float(np.asarray(radius).squeeze())
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    t = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=center[0] + radius*np.cos(t),
        y=center[1] + radius*np.sin(t),
        mode="lines",
        name=f"Prediction disk (radius={radius:.1f} mm)"
    ))
    fig.add_trace(go.Scatter(
        x=[center[0]], y=[center[1]],
        mode="markers", name="Predicted location",
        marker=dict(size=10, symbol="x")
    ))
    fig.add_trace(go.Scatter(
        x=[true[0]], y=[true[1]],
        mode="markers", name="True location",
        marker=dict(size=10)
    ))
    fig.update_layout(
        title=f"{method_name}: Single-sample uncertainty (SLA={sla_level:.2f})",
        xaxis_title="x (mm)", yaxis_title="y (mm)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    if output_path is not None:
        return write_fig(fig, output_path)
    return fig


def plot_prediction(prediction, true_position, radius, sla_level,
                     method_name="CQR-OneSided", output_path=None):
    """Plot an already-computed prediction without performing inference."""
    return plot_single_sample(
        prediction, true_position, radius, sla_level,
        method_name=method_name, output_path=output_path
    )
