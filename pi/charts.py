"""Plotly chart builders. Charts are always built from a MetricResult, so a
chart can never show a number that did not come out of the semantic layer."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#1F4E79", "#2E86AB", "#5BA4CF", "#F18F01", "#C73E1D", "#6B7A8F",
           "#3C6E71", "#A26769"]
ACCENT = "#C73E1D"
NEUTRAL = "#1F4E79"
GOOD = "#2A7F62"

LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=10, r=10, t=48, b=10),
    height=380,
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13),
    title=dict(font=dict(size=15)),
    hoverlabel=dict(font_size=12),
)


def _apply(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(**LAYOUT, title_text=title)
    return fig


def bar(result, title: str | None = None, highlight_top: bool = True) -> go.Figure:
    df = result.data.copy()
    title = title or f"{result.metric.label} by {result.dimension.replace('_', ' ')}"
    df = df.sort_values("value", ascending=True)
    worse_is_high = result.metric.direction == "lower_is_better"
    colors = [NEUTRAL] * len(df)
    if highlight_top and len(df) > 1:
        idx = df["value"].idxmax() if worse_is_high else df["value"].idxmin()
        colors = [ACCENT if i == idx else NEUTRAL for i in df.index]
    fig = go.Figure(go.Bar(
        x=df["value"], y=df["dimension"], orientation="h",
        marker_color=colors,
        text=[result.metric.fmt(v) for v in df["value"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    if result.metric.benchmark is not None:
        fig.add_vline(x=result.metric.benchmark, line_dash="dot", line_color="#888",
                      annotation_text=f"benchmark {result.metric.fmt(result.metric.benchmark)}",
                      annotation_position="top")
    fig.update_xaxes(title_text=result.metric.label)
    fig.update_yaxes(title_text="")
    return _apply(fig, title)


def line(result, title: str | None = None) -> go.Figure:
    df = result.data.copy()
    title = title or f"{result.metric.label} over time"
    period_col = "period" if "period" in df.columns else df.columns[0]
    fig = px.line(df, x=period_col, y="value", color="dimension",
                  markers=len(df[period_col].unique()) <= 12,
                  color_discrete_sequence=PALETTE)
    fig.update_traces(line_width=2.4, hovertemplate="%{x}: %{y}<extra>%{fullData.name}</extra>")
    if result.metric.benchmark is not None:
        fig.add_hline(y=result.metric.benchmark, line_dash="dot", line_color="#888",
                      annotation_text="benchmark", annotation_position="top left")
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text=result.metric.label)
    fig.update_layout(legend_title_text="")
    return _apply(fig, title)


def funnel(result, title: str = "Recruiting funnel") -> go.Figure:
    df = result.data.sort_values("stage_index")
    fig = go.Figure(go.Funnel(
        y=df["dimension"], x=df["value"],
        textinfo="value+percent initial",
        marker=dict(color=PALETTE[:len(df)]),
    ))
    return _apply(fig, title)


def stacked_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, barmode="stack",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(legend_title_text="")
    return _apply(fig, title)


def scatter_quadrant(df: pd.DataFrame, x: str, y: str, label: str,
                     title: str, x_title: str, y_title: str) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, text=label, color_discrete_sequence=[NEUTRAL])
    fig.update_traces(marker=dict(size=15, opacity=0.85), textposition="top center")
    fig.add_vline(x=df[x].mean(), line_dash="dot", line_color="#bbb")
    fig.add_hline(y=df[y].mean(), line_dash="dot", line_color="#bbb")
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return _apply(fig, title)


def diverging_bar(df: pd.DataFrame, label_col: str, value_col: str,
                  title: str, value_title: str) -> go.Figure:
    d = df.sort_values(value_col)
    colors = [ACCENT if v < 0 else GOOD for v in d[value_col]]
    fig = go.Figure(go.Bar(x=d[value_col], y=d[label_col], orientation="h",
                           marker_color=colors,
                           text=[f"{v:+.2f}" for v in d[value_col]],
                           textposition="outside", cliponaxis=False))
    fig.add_vline(x=0, line_color="#666")
    fig.update_xaxes(title_text=value_title)
    fig.update_yaxes(title_text="")
    return _apply(fig, title)


def gauge(value: float, title: str, suffix: str = "%",
          good_above: float | None = None) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": NEUTRAL},
            "steps": [
                {"range": [0, 90], "color": "#f7d7d1"},
                {"range": [90, 98], "color": "#fdecc8"},
                {"range": [98, 100], "color": "#d7ecdf"},
            ],
        },
    ))
    fig.update_layout(**{**LAYOUT, "height": 240}, title_text=title)
    return fig


def build(kind: str, result, title: str | None = None) -> go.Figure:
    kind = (kind or "").lower()
    if kind == "line":
        return line(result, title)
    if kind == "funnel":
        return funnel(result, title or "Recruiting funnel")
    if kind == "diverging":
        return diverging_bar(result.data, "dimension", "value",
                             title or result.metric.label, result.metric.label)
    return bar(result, title)


def projection(df: pd.DataFrame, title: str) -> go.Figure:
    """Baseline vs scenario headcount projection."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=df["baseline"], name="Baseline",
                             line=dict(color=NEUTRAL, width=2.4)))
    fig.add_trace(go.Scatter(x=df["month"], y=df["scenario"], name="Scenario",
                             line=dict(color=ACCENT, width=2.4, dash="dash"),
                             fill="tonexty", fillcolor="rgba(199,62,29,0.10)"))
    fig.update_xaxes(title_text="Months from reporting date")
    fig.update_yaxes(title_text="Headcount")
    fig.update_layout(legend_title_text="")
    return _apply(fig, title)
