"""

This page was entirely generate by Claude Anothropic ai based on assignment description 
and a very descriptive prompt. 

Then I went back in to customize design and layout based on what I wanted 

"""
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup & theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="LinkedIn Usage Predictor", page_icon="💼", layout="wide")

BLUE = "#0A66C2"
DARK_BLUE = "#004182"
LIGHT_BLUE = "#E8F3FF"
DARK_TEXT = "#1D2226"
MUTED_TEXT = "#5E6D77"
GREEN = "#057642"
RED = "#CC1016"
NEG_BAR = "#C2410C"   # orange for negative coefficients (validated vs. BLUE for color-blind separation)
GRID = "#E6E9EC"

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
[data-testid="stAppViewContainer"], .stApp {{ background: {BLUE}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stHeader"] * {{ color: #FFFFFF; }}
[data-testid="stExpander"] {{ background: #FFFFFF; border-radius: 14px; border: none; }}
[data-testid="stAlert"] {{ background: #FFFFFF; border-radius: 14px; }}
[data-testid="stAlert"] * {{ color: {DARK_TEXT}; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {GRID}; }}
.header {{
  background: #FFFFFF; box-shadow: 0 2px 10px rgba(0,0,0,.12);
  border-radius: 14px; padding: 22px 28px; color: {DARK_TEXT}; margin-bottom: 1.2rem;
  display: flex; align-items: center; gap: 18px;
}}
.header-icon {{
  width: 56px; height: 56px; border-radius: 12px; background: {LIGHT_BLUE};
  display: flex; align-items: center; justify-content: center; font-size: 28px; flex-shrink: 0;
}}
.header-logo {{ height: 56px; width: auto; flex-shrink: 0; }}
.header h1 {{ font-size: 1.7rem; margin: 0; color: {DARK_BLUE}; font-weight: 700; line-height: 1.2; }}
.header p {{ margin: 4px 0 0; color: {MUTED_TEXT}; font-size: .95rem; }}
.card {{
  background: #fff; border: 1px solid {GRID}; border-radius: 14px;
  padding: 22px 24px; box-shadow: 0 1px 3px rgba(0,0,0,.04); height: 100%;
}}
.card-label {{ font-size: .78rem; letter-spacing: .06em; text-transform: uppercase;
  color: {MUTED_TEXT}; font-weight: 600; margin-bottom: 10px; }}
.verdict {{ display: flex; align-items: center; gap: 12px; font-size: 1.35rem; font-weight: 700; }}
.verdict-icon {{ width: 40px; height: 40px; border-radius: 50%; color: #fff;
  display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0; }}
.prob {{ font-size: 3.2rem; font-weight: 700; color: {DARK_TEXT}; line-height: 1; margin: 18px 0 4px; }}
.prob-sub {{ color: {MUTED_TEXT}; font-size: .9rem; }}
.bar-track {{ position: relative; height: 14px; background: {LIGHT_BLUE}; border-radius: 7px; margin: 16px 0 6px; }}
.bar-fill {{ height: 100%; border-radius: 7px; }}
.bar-threshold {{ position: absolute; top: -5px; left: 50%; width: 2px; height: 24px; background: {DARK_TEXT}; opacity: .55; }}
.bar-scale {{ display: flex; justify-content: space-between; font-size: .75rem; color: {MUTED_TEXT}; }}
.profile-row {{ display: flex; justify-content: space-between; padding: 8px 0;
  border-bottom: 1px solid {GRID}; font-size: .92rem; gap: 12px; }}
.profile-row:last-child {{ border-bottom: none; }}
.profile-row span:first-child {{ color: {MUTED_TEXT}; }}
.profile-row span:last-child {{ font-weight: 600; text-align: right; }}
.section-title {{ font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 1.6rem 0 .2rem; }}
.section-sub {{ color: {MUTED_TEXT}; font-size: .9rem; margin-bottom: .6rem; }}
.disclaimer {{ color: rgba(255,255,255,.85); font-size: .78rem; text-align: center; margin-top: 2rem; }}
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
  background: {BLUE}; color: #fff; border: none; border-radius: 24px; font-weight: 600;
  padding: .6rem 1rem; width: 100%;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{ background: {DARK_BLUE}; color: #fff; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load the Part 1 model (no retraining in the app)
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"


@st.cache_resource
def load_model(path):
    obj = joblib.load(path)
    if isinstance(obj, dict):           # bundle saved by save_model.py
        return obj["model"], obj
    return obj, {}                      # plain model saved with joblib.dump(lr, ...)


if not MODEL_PATH.exists():
    st.error("**model.pkl not found.** Save your Part 1 model first: run `python save_model.py`, "
             "or run `joblib.dump(lr, 'model.pkl')` at the end of your notebook, "
             "and put model.pkl in the same folder as app.py.")
    st.stop()

model, meta = load_model(MODEL_PATH)
FEATURES = meta.get("features", list(getattr(model, "feature_names_in_",
                     ["income", "education", "parent", "married", "female", "age"])))

# ----------------------------------------------------------------------------
# Category labels — taken directly from the dataset codebook (README)
# ----------------------------------------------------------------------------
INCOME_LABELS = {
    1: "Less than $10,000",
    2: "$10,000 to under $20,000",
    3: "$20,000 to under $30,000",
    4: "$30,000 to under $40,000",
    5: "$40,000 to under $50,000",
    6: "$50,000 to under $75,000",
    7: "$75,000 to under $100,000",
    8: "$100,000 to under $150,000",
    9: "$150,000 or more",
}
EDUCATION_LABELS = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate / GED",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Four-year bachelor's degree",
    7: "Some postgraduate schooling, no degree",
    8: "Postgraduate or professional degree",
}
FEATURE_NAMES = {
    "income": "Income", "education": "Education", "parent": "Parent",
    "married": "Married", "female": "Female", "age": "Age",
}

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
LOGO_FILES = ["linkedin_logo.png", "linkedin_logo.svg", "linkedin_logo.jpg", "linkedin_logo.jpeg"]


def logo_html():
    mime = {".png": "image/png", ".svg": "image/svg+xml", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    for name in LOGO_FILES:
        f = Path(__file__).parent / name
        if f.exists():
            data = base64.b64encode(f.read_bytes()).decode()
            return f'<img class="header-logo" src="data:{mime[f.suffix.lower()]};base64,{data}" alt="Logo">'
    return '<div class="header-icon">💼</div>'


st.markdown(f"""
<div class="header">
  {logo_html()}
  <div>
    <h1>LinkedIn Usage Prediction</h1>
    <p>Machine-learning marketing insights dashboard · predict whether a person is likely to use LinkedIn</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Profile Information")
    st.caption("Describe the person. Every field is required.")

    with st.form("inputs"):
        income = st.selectbox(
            "Household income",
            options=list(INCOME_LABELS),
            format_func=lambda k: f"{k} · {INCOME_LABELS[k]}",
            index=None, placeholder="Select an income level",
            help="Dataset income scale (1–9), from the survey codebook.",
        )
        education = st.selectbox(
            "Education",
            options=list(EDUCATION_LABELS),
            format_func=lambda k: f"{k} · {EDUCATION_LABELS[k]}",
            index=None, placeholder="Select education level",
            help="Highest level of school completed (1–8), from the survey codebook.",
        )
        parent = st.radio("Parent of a child under 18 at home?", ["Yes", "No"],
                          index=None, horizontal=True)
        married = st.radio("Marital status", ["Married", "Not married"],
                           index=None, horizontal=True,
                           help="'Not married' includes divorced, separated, widowed, "
                                "never married, and living with a partner.")
        female = st.radio("Female?", ["Yes", "No"], index=None, horizontal=True)
        age = st.slider("Age (years)", min_value=18, max_value=98, value=42,
                        help="Real age, 18–98. The model treats ages above 98 as missing.")

        submitted = st.form_submit_button("Predict LinkedIn Usage")

    st.caption("Model: logistic regression (class_weight='balanced') trained in Part 1 "
               "on the social media usage survey.")

# ----------------------------------------------------------------------------
# Validate & build the model input
# ----------------------------------------------------------------------------
inputs = {"income": income, "education": education, "parent": parent,
          "married": married, "female": female}

if submitted:
    missing = [FEATURE_NAMES[k] for k, v in inputs.items() if v is None]
    if missing:
        st.warning("Please complete: " + ", ".join(missing))
        st.session_state.pop("person", None)
    else:
        st.session_state["person"] = {
            "income": int(income),
            "education": int(education),
            "parent": 1 if parent == "Yes" else 0,
            "married": 1 if married == "Married" else 0,
            "female": 1 if female == "Yes" else 0,
            "age": int(age),
        }

if "person" not in st.session_state:
    st.info("👈 Enter a person's details in the sidebar, then click **Predict LinkedIn Usage**.")
    st.markdown('<div class="disclaimer">Student machine-learning project. '
                'Not affiliated with, sponsored by, or endorsed by LinkedIn.</div>',
                unsafe_allow_html=True)
    st.stop()

person = st.session_state["person"]
person_df = pd.DataFrame([person])[FEATURES]

# ----------------------------------------------------------------------------
# Predict (classification comes from the model itself)
# ----------------------------------------------------------------------------
pred_class = int(model.predict(person_df)[0])
classes = list(model.classes_)
prob = float(model.predict_proba(person_df)[0][classes.index(1)])
pct = prob * 100

# ----------------------------------------------------------------------------
# Prediction card (full width)
# ----------------------------------------------------------------------------
if pred_class == 1:
    color, icon, verdict = GREEN, "✓", "LIKELY LINKEDIN USER"
else:
    color, icon, verdict = RED, "✕", "NOT CLASSIFIED AS A LINKEDIN USER"
st.markdown(f"""
<div class="card">
  <div class="card-label">Prediction</div>
  <div class="verdict" style="color:{color}">
    <div class="verdict-icon" style="background:{color}">{icon}</div>{verdict}
  </div>
  <div class="prob">{pct:.1f}%</div>
  <div class="prob-sub">Predicted probability of LinkedIn usage</div>
  <div class="bar-track">
    <div class="bar-fill" style="width:{pct:.1f}%; background:{BLUE};"></div>
    <div class="bar-threshold" title="Classification threshold (50%)"></div>
  </div>
  <div class="bar-scale"><span>0% · Low</span><span>50% threshold</span><span>High · 100%</span></div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Why this prediction?
# ----------------------------------------------------------------------------
coefs = pd.Series(model.coef_[0], index=FEATURES)
odds_ratio = np.exp(coefs)


def chart_layout(fig, height, xtitle=None, ytitle=None):
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color=DARK_TEXT, size=13), showlegend=False,
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=DARK_TEXT, bordercolor=GRID),
    )
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zerolinecolor="#9AA5AD", linecolor=GRID)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zerolinecolor="#9AA5AD", linecolor=GRID)
    return fig


def show_chart(fig):
    """Full-width Plotly chart; works on both newer and older Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


st.markdown('<div class="section-title">Why this prediction?</div>', unsafe_allow_html=True)
with st.expander("Open the explanation", expanded=False):
    st.markdown(
        "The model combines six characteristics — **income, education, parent status, "
        "marital status, gender, and age** — into a single score, then converts that score "
        "into a probability. People scoring above **50%** are classified as LinkedIn users."
    )

    # Coefficient chart
    order = coefs.sort_values()
    fig = go.Figure(go.Bar(
        x=order.values,
        y=[FEATURE_NAMES[f] for f in order.index],
        orientation="h",
        marker=dict(color=[BLUE if v >= 0 else NEG_BAR for v in order.values],
                    cornerradius=4),
        text=[f"{v:+.3f}" for v in order.values], textposition="outside",
        textfont=dict(color=DARK_TEXT),
        customdata=np.exp(order.values),
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:+.3f}<br>"
                      "Odds ratio: %{customdata:.2f}× per unit<extra></extra>",
    ))
    pad = (order.abs().max()) * 0.35
    fig.update_xaxes(range=[order.min() - pad, order.max() + pad])
    st.markdown("**Model coefficients** · "
                f"<span style='color:{BLUE}'>■</span> positive association &nbsp; "
                f"<span style='color:{NEG_BAR}'>■</span> negative association",
                unsafe_allow_html=True)
    show_chart(chart_layout(fig, 300, xtitle="Coefficient (change in log-odds per one-unit increase)"))


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
acc = meta.get("test_accuracy")
n = meta.get("n_rows")
model_note = (f"Model trained on {n:,} survey respondents; test accuracy {acc:.0%}. " if acc and n else "")
st.markdown(f'<div class="disclaimer">{model_note}Predictions are probabilistic estimates, not '
            'certainties.<br>Student machine-learning project. Not affiliated with, sponsored by, '
            'or endorsed by LinkedIn.</div>', unsafe_allow_html=True)
