"""Shared UI helpers: white professional theme, Times New Roman typography, KPI cards, chart styling."""
import pandas as pd
import streamlit as st

FONT = "Times New Roman"
PALETTE = ["#2E7D32", "#1565C0", "#F9A825", "#C62828", "#6A1B9A", "#00838F", "#EF6C00", "#455A64"]
GREEN_SCALE = ["#F1F8E9", "#C5E1A5", "#81C784", "#388E3C", "#1B5E20"]

CSS = """
<style>
html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, h5, h6, button, input, textarea, select,
[data-testid="stSidebar"] *, [data-testid="stMetric"] *, [data-baseweb="select"] *, [data-baseweb="tab"] *,
.stTabs button, .stAlert, .stCaption, table, th, td, .stSelectbox, .stSlider, .stNumberInput, .stRadio * {
    font-family: 'Times New Roman', Times, serif !important;
}
span[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
.stApp { background: #FFFFFF; color: #1F2933; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: rgba(255,255,255,0.9); }
h1, h2, h3, h4 { color: #1B5E20; font-weight: 700; }
p, li, label { font-size: 1.06rem; }

[data-testid="stSidebar"] { background: #F3F7F3; border-right: 1px solid #DCE6DC; }
[data-testid="stSidebar"] .stRadio label { font-size: 1.08rem; }

.page-header { border-left: 6px solid #2E7D32; background: linear-gradient(90deg, #F1F8E9 0%, #FFFFFF 100%);
    padding: 16px 22px; border-radius: 6px; margin-bottom: 18px; }
.page-header h1 { margin: 0; font-size: 2.0rem; color: #1B5E20; }
.page-header p { margin: 4px 0 0 0; color: #455A64; font-size: 1.1rem; }

.kpi { background: #FFFFFF; border: 1px solid #DCE6DC; border-top: 4px solid #2E7D32; border-radius: 8px;
    padding: 14px 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); text-align: center; height: 100%; }
.kpi-l { font-size: 0.98rem; color: #546E7A; letter-spacing: 0.3px; }
.kpi-v { font-size: 2.0rem; font-weight: 700; color: #1B5E20; line-height: 1.25; }
.kpi-s { font-size: 0.9rem; color: #78909C; }

.card { background: #FFFFFF; border: 1px solid #DCE6DC; border-radius: 8px; padding: 16px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); height: 100%; }
.card h4 { margin: 0 0 6px 0; color: #1B5E20; }
.card p { margin: 0; color: #37474F; font-size: 1.0rem; }

.flow { display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; margin: 6px 0 14px 0; }
.flow .step { flex: 1; min-width: 150px; background: #F1F8E9; border: 1px solid #C5E1A5; border-radius: 8px;
    padding: 10px 12px; text-align: center; }
.flow .step b { color: #1B5E20; display: block; font-size: 1.05rem; }
.flow .step span { font-size: 0.92rem; color: #455A64; }
.flow .arrow { align-self: center; font-size: 1.6rem; color: #2E7D32; }

table.pro-table { width: 100%; border-collapse: collapse; margin: 6px 0 14px 0; font-size: 1.0rem; }
table.pro-table th { background: #2E7D32; color: #FFFFFF; padding: 8px 10px; text-align: center; border: 1px solid #1B5E20; }
table.pro-table td { padding: 7px 10px; border: 1px solid #DCE6DC; text-align: center; color: #1F2933; }
table.pro-table tr:nth-child(even) td { background: #F6FAF6; }
.badge-ok { color: #1B5E20; font-weight: 700; }
.badge-bad { color: #C62828; font-weight: 700; }
.footer { text-align: center; color: #78909C; font-size: 0.95rem; margin-top: 40px; border-top: 1px solid #E0E7E0; padding-top: 10px; }
.stTabs [data-baseweb="tab"] { font-size: 1.08rem; }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: #2E7D32; color: #FFFFFF; border: none; border-radius: 6px; padding: 0.45rem 1.4rem; font-size: 1.05rem; }
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover { background: #1B5E20; color: #FFFFFF; }
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, sub-text)."""
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>'
                     f'<div class="kpi-s">{sub}</div></div>', unsafe_allow_html=True)
    st.write("")


def card(title: str, text: str):
    return f'<div class="card"><h4>{title}</h4><p>{text}</p></div>'


def html_table(df: pd.DataFrame, decimals: int = 3, status_col: str = None):
    """Render a DataFrame as a Times New Roman styled HTML table."""
    d = df.copy()
    if status_col and status_col in d.columns:
        d[status_col] = d[status_col].map(
            lambda v: f'<span class="{"badge-ok" if v == "Optimal" else "badge-bad"}">{v}</span>')
    html = d.to_html(index=False, border=0, classes="pro-table", float_format=lambda x: f"{x:,.{decimals}f}", escape=status_col is None)
    st.markdown(html, unsafe_allow_html=True)


def show_df(df: pd.DataFrame, height: int = 380):
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def show(fig, height: int = 420):
    """Apply the white / Times New Roman style to a Plotly figure and render it."""
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=f"{FONT}, Times, serif", size=14, color="#1F2933"),
        title=dict(font=dict(size=18, color="#1B5E20", family=f"{FONT}, Times, serif"), x=0.0, xanchor="left"),
        margin=dict(l=40, r=20, t=60, b=40), colorway=PALETTE,
        legend=dict(font=dict(size=13), bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def footer():
    st.markdown('<div class="footer">Smart Agriculture Analytics and Crop Yield Prediction System &nbsp;|&nbsp; '
                'Data Warehousing and Data Mining Capstone &nbsp;|&nbsp; Team No. 5</div>', unsafe_allow_html=True)
