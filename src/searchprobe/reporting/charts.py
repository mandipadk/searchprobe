"""Chart generation for benchmark reports using Plotly."""

from typing import Any

import numpy as np
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
                else f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
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


# --- New Visualization Functions for Research Modules ---


def create_vulnerability_heatmap(
    data: dict[str, dict[str, float]],
    title: str = "Embedding Vulnerability by Model and Category",
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """Create a vulnerability heatmap (model x category).

    Args:
        data: Dict of model -> category -> vulnerability_score
        title: Chart title
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure
    """
    if not data:
        return go.Figure()

    models = list(data.keys())
    all_categories = set()
    for model_data in data.values():
        all_categories.update(model_data.keys())
    categories = sorted(all_categories)

    z = []
    for model in models:
        row = [data[model].get(cat, 0) for cat in categories]
        z.append(row)

    # Custom text annotations
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[c.replace("_", " ").title() for c in categories],
            y=models,
            colorscale=[
                [0, "#2ecc71"],     # Green (low vulnerability)
                [0.3, "#f1c40f"],   # Yellow
                [0.6, "#e67e22"],   # Orange
                [1, "#e74c3c"],     # Red (high vulnerability)
            ],
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate=(
                "Model: %{y}<br>Category: %{x}<br>"
                "Vulnerability: %{z:.3f}<extra></extra>"
            ),
            colorbar=dict(title="Vulnerability", tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Adversarial Category",
        yaxis_title="Embedding Model",
        height=height,
        width=width,
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_similarity_distribution(
    adversarial_sims: list[float],
    baseline_sims: list[float],
    random_sims: list[float],
    title: str = "Similarity Distribution: Adversarial vs Baseline vs Random",
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """Create overlapping violin plots for similarity distributions.

    Args:
        adversarial_sims: Adversarial pair similarities
        baseline_sims: Same-topic baseline similarities
        random_sims: Random pair similarities
        title: Chart title
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    datasets = [
        ("Random Pairs", random_sims, "#3498db"),
        ("Baseline (Same Topic)", baseline_sims, "#2ecc71"),
        ("Adversarial Pairs", adversarial_sims, "#e74c3c"),
    ]

    for name, sims, color in datasets:
        if sims:
            fig.add_trace(
                go.Violin(
                    y=sims,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    line_color=color,
                    fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.25)",
                    opacity=0.7,
                )
            )

    fig.update_layout(
        title=title,
        yaxis_title="Cosine Similarity",
        height=height,
        width=width,
        showlegend=True,
    )

    return fig


def create_embedding_projection(
    embeddings_2d: list[tuple[float, float]],
    labels: list[str],
    categories: list[str],
    title: str = "Embedding Space Projection",
    method: str = "t-SNE",
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """Create a 2D embedding projection colored by category.

    Args:
        embeddings_2d: List of (x, y) coordinates from t-SNE/UMAP
        labels: Text labels for each point
        categories: Category for each point (used for coloring)
        title: Chart title
        method: Projection method name (for display)
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure
    """
    if not embeddings_2d:
        return go.Figure()

    fig = go.Figure()

    # Group by category
    unique_categories = sorted(set(categories))
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        "#1f77b4", "#d62728", "#2ca02c",
    ]

    for i, cat in enumerate(unique_categories):
        mask = [j for j, c in enumerate(categories) if c == cat]
        xs = [embeddings_2d[j][0] for j in mask]
        ys = [embeddings_2d[j][1] for j in mask]
        texts = [labels[j] for j in mask]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=cat.replace("_", " ").title(),
                text=texts,
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                marker=dict(
                    color=colors[i % len(colors)],
                    size=8,
                    opacity=0.7,
                ),
            )
        )

    fig.update_layout(
        title=f"{title} ({method})",
        xaxis_title=f"{method} Dimension 1",
        yaxis_title=f"{method} Dimension 2",
        height=height,
        width=width,
        showlegend=True,
    )

    return fig


def create_sensitivity_map_chart(
    query: str,
    word_scores: dict[str, float],
    title: str = "Query Sensitivity Map",
    height: int = 300,
    width: int = 800,
) -> go.Figure:
    """Create a word-level importance visualization (sensitivity map).

    Args:
        query: Original query text
        word_scores: Word -> sensitivity score [0, 1]
        title: Chart title
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure
    """
    if not word_scores:
        return go.Figure()

    words = query.split()
    scores = [word_scores.get(word, 0.0) for word in words]

    # Color scale from green (low) to red (high sensitivity)
    colors = [
        f"rgb({int(255 * s)}, {int(255 * (1 - s))}, 50)"
        for s in scores
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=words,
                y=scores,
                marker_color=colors,
                text=[f"{s:.2f}" for s in scores],
                textposition="outside",
                hovertemplate="Word: %{x}<br>Sensitivity: %{y:.3f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"{title}: \"{query}\"",
        xaxis_title="Word",
        yaxis_title="Sensitivity Score",
        yaxis=dict(range=[0, 1.2]),
        height=height,
        width=width,
    )

    return fig


def create_embedding_gap_chart(
    data: dict[str, float],
    title: str = "NDCG Improvement Potential by Category",
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """Create a bar chart showing NDCG improvement potential per category.

    Shows how much a cross-encoder reranking improves results, indicating
    the "embedding gap" for each adversarial category.

    Args:
        data: Dict of category -> mean NDCG improvement
        title: Chart title
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure
    """
    if not data:
        return go.Figure()

    # Sort by improvement
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0].replace("_", " ").title() for item in sorted_items]
    improvements = [item[1] for item in sorted_items]

    # Color by severity
    colors = []
    for imp in improvements:
        if imp >= 0.3:
            colors.append("#e74c3c")  # Severe
        elif imp >= 0.15:
            colors.append("#e67e22")  # Significant
        elif imp >= 0.05:
            colors.append("#f1c40f")  # Moderate
        else:
            colors.append("#2ecc71")  # Minimal

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=improvements,
                marker_color=colors,
                text=[f"{v:.1%}" for v in improvements],
                textposition="outside",
                hovertemplate="Category: %{x}<br>NDCG Improvement: %{y:.3f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Adversarial Category",
        yaxis_title="NDCG Improvement (Cross-Encoder Reranking)",
        yaxis=dict(tickformat=".0%"),
        height=height,
        width=width,
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_3d_embedding_explorer(
    embeddings_3d: list[tuple[float, float, float]],
    labels: list[str],
    categories: list[str],
    title: str = "3D Embedding Explorer",
    height: int = 800,
    width: int = 1000,
) -> go.Figure:
    """Create an interactive 3D scatter plot for embedding exploration.

    Args:
        embeddings_3d: List of (x, y, z) coordinates
        labels: Text labels for each point
        categories: Category for each point
        title: Chart title
        height: Chart height
        width: Chart width

    Returns:
        Plotly Figure
    """
    if not embeddings_3d:
        return go.Figure()

    fig = go.Figure()

    unique_categories = sorted(set(categories))
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        "#1f77b4", "#d62728", "#2ca02c",
    ]

    for i, cat in enumerate(unique_categories):
        mask = [j for j, c in enumerate(categories) if c == cat]
        xs = [embeddings_3d[j][0] for j in mask]
        ys = [embeddings_3d[j][1] for j in mask]
        zs = [embeddings_3d[j][2] for j in mask]
        texts = [labels[j] for j in mask]

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name=cat.replace("_", " ").title(),
                text=texts,
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                marker=dict(
                    color=colors[i % len(colors)],
                    size=5,
                    opacity=0.7,
                ),
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
        ),
        height=height,
        width=width,
        showlegend=True,
    )

    return fig
