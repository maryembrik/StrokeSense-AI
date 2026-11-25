import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import streamlit as st

from components.core import predictor, styles
from components.visuals import svg_img

st.set_page_config(page_title="StrokeSense AI", page_icon=str(HERE / "assets/logo.svg"), layout="wide", initial_sidebar_state="auto")
styles()
PAGES = [("overview", "Overview", "radar"), ("simulator", "Patient Simulator", "tune"), ("detective", "Leak Detective", "search"),
         ("lab", "Research Lab", "science"), ("threshold", "Threshold Studio", "vertical_align_center"), ("drift", "Drift Observatory", "monitoring")]
pages = [st.Page(HERE / f"views/{v}.py", title=t, icon=f":material/{i}:", url_path=None if n == 0 else v, default=n == 0) for n, (v, t, i) in enumerate(PAGES)]
current = st.navigation(pages, position="hidden")

with st.sidebar:
    logo = svg_img((HERE / "assets/logo.svg").read_text(), "StrokeSense AI logo")
    st.html(f'<div class="brand">{logo}<div><b>StrokeSense AI</b><span>The Science of Seeing Risk.</span></div></div>')
    for p in pages:
        with st.container(key="navactive" if p == current else f"nav_{p.title}"):
            st.page_link(p)
    try:
        version = predictor().version[:8]
    except Exception:
        version = "unavailable"
    st.html(f'<div class="meta"><div>Model<b>{version}</b></div><div>Evaluation<b>4 × 5-fold CV</b></div><div>Test set<b>scored once</b></div>'
            "<p>Academic research demonstration, 2026. Not a clinical system.</p></div>")
try:
    predictor()
except Exception as e:
    st.error(f"The frozen model could not be loaded, so the app has stopped instead of substituting another one. {e}")
    st.stop()
current.run()
