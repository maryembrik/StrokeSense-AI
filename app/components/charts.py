"""Plotly design system. Colour follows the entity on every page."""
import plotly.graph_objects as go
import streamlit as st

TEAL, CYAN, AMBER, CORAL, WHITE, MUTE, SLATE = "#25C2B1", "#70D9D0", "#D9A45B", "#E0786B", "#F3F5F2", "#A3B4C4", "#6F8599"
INK, DTEAL, DAMBER = "#14222F", "#0B8578", "#A36A1C"  # darker steps for the light surface
# one fixed colour and dash per model; the two highlighted models are teal and amber
STYLE = {"Logistic 4 (selected)": (TEAL, "solid"), "Age only": (AMBER, "dash"), "Logistic": ("#9DB6F5", "solid"), "Logistic 9": ("#9DB6F5", "dot"), "CatBoost": ("#C79BE8", "solid"),
         "EBM monotone": ("#C79BE8", "dot"), "HGB monotone": ("#8FA6BA", "solid"), "XGBoost": ("#8FA6BA", "dot"), "Balanced RF": ("#E9A6C0", "solid"), "Random Forest": ("#E9A6C0", "dot")}


def rgba(h, a):
    return f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"


def fig():
    return go.Figure()


def show(f, key, h=380, paper=False, legend=True, x="", y="", xfmt=None, yfmt=None, unified=False, **xy):
    ink, grid, mute = (INK, "rgba(20,34,47,.10)", "#465868") if paper else (WHITE, "rgba(163,180,196,.13)", MUTE)
    f.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=ink),
                    margin=dict(l=8, r=18, t=34 if legend else 12, b=8), showlegend=legend, transition=dict(duration=450, easing="cubic-in-out"),
                    legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=12.5, color=ink), itemsizing="constant"),
                    hovermode="x unified" if unified else "closest",
                    hoverlabel=dict(bgcolor="#0C1624", bordercolor="rgba(112,217,208,.45)", font=dict(family="JetBrains Mono, Consolas, monospace", size=12, color=WHITE)))
    ax = dict(gridcolor=grid, zeroline=False, automargin=True, tickfont=dict(family="JetBrains Mono, Consolas, monospace", size=11.5, color=mute), title_font=dict(size=12.5, color=mute))
    if "mr" in xy:
        f.update_layout(margin_r=xy["mr"])
    f.update_xaxes(title=x, **ax)
    f.update_yaxes(title=y, **ax)
    if xfmt:
        f.update_xaxes(tickformat=xfmt)
    if yfmt:
        f.update_yaxes(tickformat=yfmt)
    if "xr" in xy:
        f.update_xaxes(range=xy["xr"])
    if "yr" in xy:
        f.update_yaxes(range=xy["yr"])
    st.plotly_chart(f, width="stretch", theme=None, key=key, config={"displayModeBar": False, "responsive": True})


def line(f, x, y, name, color, dash="solid", width=2.5, fill=None, hover="%{y:.1%}", **k):
    f.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, dash=dash, width=width), fill="tozeroy" if fill else None,
                           fillcolor=rgba(color, fill) if fill else None, hovertemplate=hover + f"<extra>{name}</extra>", **k))


def points(f, x, y, name, color, symbol="circle", size=13, lo=None, hi=None, horizontal=False, hover=None, text=None, ring="#17283A", **k):
    e = {}
    if lo is not None:
        v = x if horizontal else y
        err = dict(type="data", symmetric=False, array=[b - a for a, b in zip(v, hi)], arrayminus=[a - b for a, b in zip(v, lo)], color=color, thickness=2, width=5)
        e = {"error_x" if horizontal else "error_y": err}
    f.add_trace(go.Scatter(x=x, y=y, name=name, mode="markers+text" if text is not None else "markers", text=text, textposition="middle right",
                           textfont=dict(family="JetBrains Mono, Consolas, monospace", size=12), marker=dict(size=size, color=color, symbol=symbol, line=dict(color=ring, width=2)),
                           hovertemplate=(hover or "%{x}, %{y}") + "<extra></extra>", **e, **k))


def vline(f, x, text="", color=MUTE, dash="dot", pos="top right"):
    f.add_vline(x=x, line=dict(color=color, width=1.5, dash=dash), annotation_text=text, annotation_position=pos, annotation_font=dict(size=12, color=color))


def hline(f, y, text="", color=MUTE, dash="dot", pos="top left"):
    f.add_hline(y=y, line=dict(color=color, width=1.5, dash=dash), annotation_text=text, annotation_position=pos, annotation_font=dict(size=12, color=color))
