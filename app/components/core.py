"""Data access and layout primitives. Every displayed metric is read from artifacts/ through this module."""
import json
import sys
from contextlib import contextmanager
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import streamlit as st

from src.inference import Predictor

MAIN, BASE = "Logistic 4 (selected)", "Age only"
NAMES = {MAIN: "Logistic, 4 features", "Logistic": "Logistic, tuned", "Logistic 9": "Logistic, 9 features", "CatBoost": "CatBoost", "EBM monotone": "EBM, monotone",
         "HGB monotone": "Monotone boosting", "XGBoost": "XGBoost", "Balanced RF": "Balanced random forest", "Random Forest": "Random forest", BASE: "Age only"}
BAND = {"0-40": "40 or under", "41-60": "41–60", "61-70": "61–70", "71+": "71+"}
MIN_CASES = 10  # below this, a subgroup rate is not shown
DISCLAIMER = "This educational research application classifies observations according to the stroke label available in the dataset. It has not been validated as a prospective future-stroke prediction tool or a clinical risk calculator. It is not a medical diagnosis or a substitute for professional medical assessment."


@st.cache_resource
def predictor():
    return Predictor()


@st.cache_data
def csv(name, index_col=None):
    return pd.read_csv(ROOT / "artifacts/results" / name, index_col=index_col)


@st.cache_data
def research():
    return json.load(open(ROOT / "artifacts/app/research.json", encoding="utf-8"))


def results():
    return csv("FINAL_RESULTS.csv", 0)


def final_test():
    """The single scoring of the locked test split. Never pooled with cross-validation figures."""
    return json.load(open(ROOT / "artifacts/results/test_results.json", encoding="utf-8"))


def subgroups():
    return pd.concat([csv("subgroups_verified.csv"), pd.DataFrame(research()["residence"])]).set_index("group")


def calib():
    return csv("calibration_outer.csv", 0)


def age_gain(window=3, weighting="pair"):
    d = csv("age_matched_sensitivity.csv")
    return d[(d.window == window) & (d.weighting == weighting)].iloc[0]


# ---------- layout
def styles():
    st.html(APP / "assets/styles.css")


def html(s):
    st.html(s)


def hero(eyebrow, title, lead):
    html(f'<div class="eyebrow">{eyebrow}</div><h1 class="display">{title}</h1><p class="hero-lead">{lead}</p>')


def section(title, lead=""):
    html(f'<div class="sec"><h2>{title}</h2>' + (f"<p>{lead}</p>" if lead else "") + "</div>")


def chart_title(t, sub=""):
    html(f'<p class="ct">{t}</p>' + (f'<p class="cs">{sub}</p>' if sub else ""))


@contextmanager
def panel(key, paper=False):
    with st.container(key=("paper_" if paper else "panel_") + key):
        yield


def odometer(v):
    """Digits roll into place; the real value is exposed to assistive technology."""
    cells = []
    for c in v:
        if c.isdigit():
            n = int(c)
            cells.append(f'<span class="od"><i style="transform:translateY(-{n}em)">' + "<br>".join("0123456789"[: n + 1]) + "</i></span>")
        else:
            cells.append(escape(c))
    return f'<span class="sr">{escape(v)}</span><span aria-hidden="true">{"".join(cells)}</span>'


def kpis(items, small=False):
    html(f'<div class="kpis{" n" if small else ""}">' + "".join(
        f'<div class="kpi"><div class="v{"" if any(ch.isdigit() for ch in v) else " t"}">{odometer(v) if not small else escape(v)}</div><div class="l">{l}</div></div>' for v, l in items) + "</div>")


def callout(t, tone=""):
    html(f'<div class="callout {tone}">{t}</div>')


def insight(t):
    html(f'<p class="insight">{t}</p>')


def warn(t):
    html(f'<div class="warn" role="note"><div>{t}</div></div>')


def table(df, highlight=None, tag="primary", wrap=False):
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        hi = highlight is not None and r.iloc[0] == highlight
        cells = [escape(str(v)) for v in r]
        if hi:
            cells[0] += f'<span class="tag">{tag}</span>'
        body += f'<tr class="{"hi" if hi else ""}">' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    html(f'<div class="tbl{" wrap" if wrap else ""}" tabindex="0"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def link(view, label, icon=None):
    try:
        st.page_link(f"views/{view}.py", label=label, icon=icon)
    except Exception:  # view executed on its own in tests: no page registry
        st.caption(label)


def disclaimer():
    html(f'<div class="disc">{DISCLAIMER}</div>')
