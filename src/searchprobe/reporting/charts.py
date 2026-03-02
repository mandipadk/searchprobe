"""Chart generation for benchmark reports using Plotly."""

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_radar_chart(
    data: dict[str, dict[str, float]],
    title: str = "Provider Performance by Category",
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """Create a radar chart comparing providers across categories.

    Args:
        data: Dict of provider -> category -> score
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly Figure object
    """
    if not data:
        return go.Figure()

    # Get all categories
    all_categories = set()
    for provider_data in data.values():
        all_categories.update(provider_data.keys())
    categories = sorted(all_categories)

    # Close the radar by repeating first category
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    # Color palette
    colors = [
        "#636EFA",  # Blue
        "#EF553B",  # Red
        "#00CC96",  # Green
        "#AB63FA",  # Purple
        "#FFA15A",  # Orange
        "#19D3F3",  # Cyan
    ]

    for i, (provider, scores) in enumerate(data.items()):
        # Get scores in category order, closing the loop
        values = [scores.get(cat, 0) for cat in categories]
        values_closed = values + [values[0]]

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=provider,
                line=dict(color=color),
                fillcolor=color.replace(")", ", 0.2)").replace("rgb", "rgba")
                if "rgb" in color
                else color + "33",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            )
        ),
        showlegend=True,
        title=title,
        height=height,
        width=width,
    )

    return fig


def create_heatmap(
    data: dict[str, dict[str, float]],
    title: str = "Score Heatmap",
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """Create a heatmap of scores by provider and category.

    Args:
        data: Dict of provider -> category -> score
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Plotly Figure object
    """
    if not data:
        return go.Figure()

    # Get all categories
    all_categories = set()
    for provider_data in data.values():
        all_categories.update(provider_data.keys())
    categories = sorted(all_categories)

    providers = list(data.keys())

    # Build matrix
    z = []
    for provider in providers:
        row = [data[provider].get(cat, 0) for cat in categories]
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=categories,
            y=providers,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="Provider: %{y}<br>Category: %{x}<br>Score: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis_title="Provider",
        height=height,
        width=width,
    )

    return fig


def create_bar_chart(
    data: dict[str, float],
    title: str = "Overall Scores",
    x_label: str = "Provider",
    y_label: str = "Score",
    height: int = 400,
    width: int = 600,
    error_bars: dict[str, tuple[float, float]] | None = None,
) -> go.Figure:
    """Create a bar chart of scores.

    Args:
        data: Dict of label -> score
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        height: Chart height
        width: Chart width
        error_bars: Optional dict of label -> (lower_error, upper_error)

    Returns:
        Plotly Figure object
    """
    if not data:
        return go.Figure()

    labels = list(data.keys())
    values = list(data.values())

    # Calculate error bar values if provided
    error_y_dict: dict[str, Any] = {}
    if error_bars:
        error_minus = []
        error_plus = []
        for label in labels:
            if label in error_bars:
                lower, upper = error_bars[label]
                mean = data[label]
                error_minus.append(mean - lower)
                error_plus.append(upper - mean)
            else:
                error_minus.append(0)
                error_plus.append(0)

        error_y_dict = dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
        )

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color="#636EFA",
                error_y=error_y_dict if error_bars else None,
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis=dict(range=[0, 1]),
        height=height,
        width=width,
    )

    return fig


def create_failure_mode_chart(
    data: dict[str, int],
    title: str = "Failure Mode Distribution",
    top_n: int = 10,
    height: int = 400,
    width: int = 600,
) -> go.Figure:
    """Create a horizontal bar chart of failure modes.

    Args:
        data: Dict of failure_mode -> count
        title: Chart title
        top_n: Number of top failure modes to show
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure object
    """
    if not data:
        return go.Figure()

    # Sort and take top N
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Reverse for horizontal bar chart (top at top)
    modes = [item[0] for item in reversed(sorted_items)]
    counts = [item[1] for item in reversed(sorted_items)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=modes,
                orientation="h",
                marker_color="#EF553B",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Count",
        yaxis_title="Failure Mode",
        height=height,
        width=width,
    )

    return fig


def create_latency_comparison(
    data: dict[str, list[float]],
    title: str = "Latency Distribution by Provider",
    height: int = 400,
    width: int = 700,
) -> go.Figure:
    """Create a box plot comparing latency distributions.

    Args:
        data: Dict of provider -> list of latency values (ms)
        title: Chart title
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure object
    """
    if not data:
        return go.Figure()

    fig = go.Figure()

    for provider, latencies in data.items():
        fig.add_trace(
            go.Box(
                y=latencies,
                name=provider,
                boxpoints="outliers",
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title="Latency (ms)",
        height=height,
        width=width,
    )

    return fig


def create_cost_breakdown(
    data: dict[str, float],
    title: str = "Cost Breakdown by Provider",
    height: int = 400,
    width: int = 500,
) -> go.Figure:
    """Create a pie chart of cost breakdown.

    Args:
        data: Dict of provider -> total cost
        title: Chart title
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure object
    """
    if not data:
        return go.Figure()

    labels = list(data.keys())
    values = list(data.values())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo="label+percent",
                textposition="inside",
                hovertemplate="%{label}: $%{value:.4f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=height,
        width=width,
    )

    return fig


def create_multi_chart_figure(
    charts: list[tuple[go.Figure, str]],
    title: str = "Benchmark Results",
    cols: int = 2,
) -> go.Figure:
    """Combine multiple charts into a single figure.

    Args:
        charts: List of (figure, subplot_title) tuples
        title: Overall figure title
        cols: Number of columns

    Returns:
        Combined Plotly Figure
    """
    if not charts:
        return go.Figure()

    rows = (len(charts) + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[t for _, t in charts],
        specs=[[{"type": "xy"} for _ in range(cols)] for _ in range(rows)],
    )

    for i, (chart, _) in enumerate(charts):
        row = i // cols + 1
        col = i % cols + 1

        for trace in chart.data:
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        title=title,
        height=400 * rows,
        showlegend=True,
    )

    return fig
