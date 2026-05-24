from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageEnhance, ImageFilter
from pytesseract import Output

try:
    import fitz  # PyMuPDF
except ImportError:  # PDF image rendering is optional.
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


APP_NAME = "FinOCR AI"
APP_TAGLINE = "Autonomous document intelligence for BFSI teams"
DEFAULT_TESSERACT = r"C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
DEFAULT_TESSDATA = r"C:\Users\hp\AppData\Local\Programs\Tesseract-OCR\tessdata"

LANGUAGE_OPTIONS = {
    "English": "eng",
    "Hindi + English": "hin+eng",
    "Marathi + English": "mar+eng",
    "Tamil + English": "tam+eng",
    "Telugu + English": "tel+eng",
    "Arabic + English": "ara+eng",
}

DOCUMENT_KEYWORDS = {
    "Bank Statement": ["statement", "account", "ifsc", "transaction", "withdrawal", "deposit", "balance"],
    "Invoice": ["invoice", "gst", "bill to", "tax", "total", "subtotal", "qty"],
    "Payslip": ["salary", "earnings", "deductions", "net pay", "employee", "provident"],
    "Loan Document": ["loan", "emi", "interest", "principal", "tenure", "sanction"],
    "Insurance": ["policy", "premium", "insured", "claim", "coverage", "nominee"],
    "Tax Document": ["income tax", "pan", "assessment", "tds", "form 16", "deduction"],
}

SAMPLE_TRANSACTIONS = pd.DataFrame(
    {
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Income": [142000, 138500, 151200, 149900, 158700, 166300],
        "Expenses": [91300, 88400, 97800, 101200, 94200, 107500],
        "Risk Events": [4, 3, 7, 5, 2, 6],
    }
)


@dataclass
class OCRResult:
    text: str
    confidence: float
    word_count: int
    line_count: int
    detected_type: str


def configure_tesseract() -> None:
    tesseract_cmd = os.getenv("TESSERACT_CMD", DEFAULT_TESSERACT)
    tessdata_prefix = os.getenv("TESSDATA_PREFIX", DEFAULT_TESSDATA)

    if os.path.exists(tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if os.path.isdir(tessdata_prefix):
        os.environ.setdefault("TESSDATA_PREFIX", tessdata_prefix)


def init_state() -> None:
    defaults = {
        "theme": "Dark",
        "active_page": "Command Center",
        "notifications": [
            "KYC extractor ready",
            "Fraud rules updated",
            "Analytics workspace synced",
        ],
        "chat_history": [
            {
                "role": "assistant",
                "content": "Hi, I am your FinOCR assistant. Upload a document or ask about OCR, fraud checks, loans, or financial insights.",
            }
        ],
        "latest_ocr_text": "",
        "profile": {"name": "BFSI Analyst", "role": "Credit Operations", "workspace": "FinOCR AI"},
        "signed_in": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_css(theme: str) -> None:
    if theme == "Light":
        palette = {
            "bg0": "#f5f8ff",
            "bg1": "#eaf1ff",
            "surface": "rgba(255, 255, 255, 0.76)",
            "surface2": "rgba(255, 255, 255, 0.55)",
            "text": "#101828",
            "muted": "#53627d",
            "border": "rgba(38, 57, 96, 0.15)",
            "sidebar": "rgba(255,255,255,0.82)",
        }
    else:
        palette = {
            "bg0": "#07111f",
            "bg1": "#110f26",
            "surface": "rgba(13, 26, 47, 0.72)",
            "surface2": "rgba(255, 255, 255, 0.08)",
            "text": "#eef6ff",
            "muted": "#9fb1cf",
            "border": "rgba(255, 255, 255, 0.14)",
            "sidebar": "rgba(8, 15, 29, 0.82)",
        }

    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            :root {{
                --bg0: {palette["bg0"]};
                --bg1: {palette["bg1"]};
                --surface: {palette["surface"]};
                --surface2: {palette["surface2"]};
                --text: {palette["text"]};
                --muted: {palette["muted"]};
                --border: {palette["border"]};
                --sidebar: {palette["sidebar"]};
                --cyan: #2ee9ff;
                --purple: #8a5cff;
                --blue: #1a74ff;
                --green: #3ee58f;
                --gold: #f7c948;
                --coral: #ff6b7a;
                --shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
                --radius: 8px;
            }}

            html, body, [class*="css"] {{
                font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                letter-spacing: 0;
            }}

            .stApp {{
                color: var(--text);
                background:
                    radial-gradient(circle at 10% 6%, rgba(46, 233, 255, 0.18), transparent 26rem),
                    radial-gradient(circle at 88% 12%, rgba(138, 92, 255, 0.20), transparent 30rem),
                    linear-gradient(135deg, var(--bg0), var(--bg1) 48%, #071827);
                background-attachment: fixed;
            }}

            .stApp::before {{
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                opacity: 0.28;
                background-image:
                    linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
                background-size: 44px 44px;
                mask-image: linear-gradient(to bottom, black, transparent 88%);
            }}

            #MainMenu, footer, header {{ visibility: hidden; }}
            .block-container {{
                max-width: 1220px;
                padding-top: 1.4rem;
                padding-bottom: 3.5rem;
            }}

            [data-testid="stSidebar"] {{
                background: var(--sidebar);
                border-right: 1px solid var(--border);
                backdrop-filter: blur(26px);
                box-shadow: 12px 0 40px rgba(0, 0, 0, 0.18);
            }}

            [data-testid="stSidebar"] * {{
                color: var(--text);
            }}

            [data-testid="stSidebar"] .stRadio > div {{
                gap: 0.35rem;
            }}

            [data-testid="stSidebar"] label[data-baseweb="radio"] {{
                background: rgba(255,255,255,0.06);
                border: 1px solid transparent;
                border-radius: var(--radius);
                padding: 0.72rem 0.85rem;
                transition: transform .24s ease, border-color .24s ease, background .24s ease;
            }}

            [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{
                transform: translateX(4px);
                border-color: rgba(46, 233, 255, 0.34);
                background: rgba(46, 233, 255, 0.10);
            }}

            h1, h2, h3, h4, h5, h6, p, label, span {{
                color: var(--text);
            }}

            p, .stMarkdown, [data-testid="stMarkdownContainer"] {{
                color: var(--muted);
            }}

            .app-shell {{
                animation: pageIn .55s ease both;
            }}

            @keyframes pageIn {{
                from {{ opacity: 0; transform: translateY(12px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}

            .brand-card, .glass, .feature-card, .metric-card, .module-card, .insight-panel {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                backdrop-filter: blur(24px);
            }}

            .brand-card {{
                padding: 1rem;
                margin-bottom: 1rem;
            }}

            .brand-title {{
                font-size: 1.2rem;
                font-weight: 800;
                margin: 0;
                background: linear-gradient(90deg, var(--cyan), #ffffff, var(--green));
                -webkit-background-clip: text;
                color: transparent;
            }}

            .brand-subtitle {{
                color: var(--muted);
                font-size: 0.78rem;
                margin: 0.2rem 0 0;
            }}

            .status-row {{
                display: grid;
                gap: .55rem;
                margin: .9rem 0 1rem;
            }}

            .status-pill {{
                align-items: center;
                background: rgba(46, 233, 255, .08);
                border: 1px solid rgba(46, 233, 255, .18);
                border-radius: var(--radius);
                display: flex;
                justify-content: space-between;
                padding: .58rem .72rem;
                color: var(--muted);
                font-size: .78rem;
            }}

            .pulse-dot {{
                width: .55rem;
                height: .55rem;
                border-radius: 50%;
                background: var(--green);
                box-shadow: 0 0 0 rgba(62, 229, 143, .52);
                animation: pulse 1.8s infinite;
            }}

            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(62, 229, 143, .56); }}
                70% {{ box-shadow: 0 0 0 10px rgba(62, 229, 143, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(62, 229, 143, 0); }}
            }}

            .hero {{
                display: grid;
                grid-template-columns: minmax(0, 1fr) minmax(320px, 0.74fr);
                gap: 1.2rem;
                align-items: stretch;
                margin-bottom: 1.2rem;
            }}

            .hero-copy {{
                padding: clamp(1.25rem, 3vw, 2.4rem);
                min-height: 440px;
                position: relative;
                overflow: hidden;
            }}

            .eyebrow {{
                display: inline-flex;
                align-items: center;
                gap: .45rem;
                padding: .42rem .7rem;
                border-radius: 999px;
                background: rgba(46,233,255,.10);
                border: 1px solid rgba(46,233,255,.24);
                color: var(--cyan);
                font-weight: 700;
                font-size: .78rem;
                text-transform: uppercase;
            }}

            .hero h1 {{
                margin: 1rem 0 .7rem;
                font-size: clamp(2.4rem, 7vw, 5.4rem);
                line-height: .96;
                font-weight: 800;
                letter-spacing: 0;
                color: var(--text);
            }}

            .gradient-text {{
                background: linear-gradient(92deg, #ffffff 5%, var(--cyan) 38%, var(--purple) 70%, var(--green));
                -webkit-background-clip: text;
                color: transparent;
                animation: shimmer 7s ease infinite;
                background-size: 220% auto;
            }}

            @keyframes shimmer {{
                0%, 100% {{ background-position: 0% center; }}
                50% {{ background-position: 100% center; }}
            }}

            .hero-lede {{
                max-width: 680px;
                font-size: 1.05rem;
                line-height: 1.7;
                color: var(--muted);
            }}

            .hero-actions {{
                display: flex;
                flex-wrap: wrap;
                gap: .75rem;
                margin-top: 1.3rem;
            }}

            .hero-button {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 42px;
                padding: .75rem 1rem;
                border-radius: var(--radius);
                font-weight: 800;
                text-decoration: none;
                transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease;
            }}

            .hero-button.primary {{
                color: #04111f;
                background: linear-gradient(135deg, var(--cyan), var(--green));
                box-shadow: 0 12px 34px rgba(46, 233, 255, .25);
            }}

            .hero-button.secondary {{
                color: var(--text);
                border: 1px solid rgba(255,255,255,.18);
                background: rgba(255,255,255,.08);
            }}

            .hero-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 18px 44px rgba(138, 92, 255, .24);
            }}

            .ai-visual {{
                min-height: 440px;
                position: relative;
                overflow: hidden;
                padding: 1.2rem;
                background:
                    linear-gradient(135deg, rgba(26,116,255,.18), rgba(138,92,255,.14)),
                    var(--surface);
            }}

            .ai-frame {{
                height: 100%;
                min-height: 390px;
                border-radius: var(--radius);
                border: 1px solid rgba(255,255,255,.15);
                background:
                    linear-gradient(rgba(46,233,255,.08) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(46,233,255,.08) 1px, transparent 1px),
                    rgba(2, 9, 23, .36);
                background-size: 30px 30px;
                position: relative;
                overflow: hidden;
            }}

            .scan-document {{
                position: absolute;
                left: 12%;
                right: 12%;
                top: 17%;
                height: 62%;
                border-radius: var(--radius);
                background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(224,242,254,.84));
                color: #0b1220;
                box-shadow: 0 28px 80px rgba(0,0,0,.32);
                transform: perspective(900px) rotateX(8deg) rotateY(-8deg);
            }}

            .doc-line {{
                height: 9px;
                border-radius: 999px;
                background: rgba(15, 23, 42, .18);
                margin: 18px 22px;
            }}

            .doc-line.short {{ width: 46%; }}
            .doc-line.medium {{ width: 66%; }}
            .doc-line.long {{ width: 78%; }}

            .scan-line {{
                position: absolute;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, transparent, var(--cyan), var(--green), transparent);
                box-shadow: 0 0 22px rgba(46,233,255,.85);
                animation: scan 3.2s ease-in-out infinite;
            }}

            @keyframes scan {{
                0%, 100% {{ top: 18%; opacity: .25; }}
                50% {{ top: 76%; opacity: 1; }}
            }}

            .neural-node {{
                position: absolute;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: var(--cyan);
                box-shadow: 0 0 24px var(--cyan);
                animation: floatNode 4.5s ease-in-out infinite;
            }}

            .node-1 {{ left: 12%; top: 10%; }}
            .node-2 {{ right: 14%; top: 18%; animation-delay: .7s; background: var(--green); }}
            .node-3 {{ left: 20%; bottom: 14%; animation-delay: 1.1s; background: var(--gold); }}
            .node-4 {{ right: 24%; bottom: 11%; animation-delay: 1.5s; background: var(--purple); }}

            @keyframes floatNode {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-12px); }}
            }}

            .ai-readout {{
                position: absolute;
                left: 1rem;
                right: 1rem;
                bottom: 1rem;
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: .6rem;
            }}

            .readout-card {{
                padding: .7rem;
                border-radius: var(--radius);
                background: rgba(5, 12, 26, .72);
                border: 1px solid rgba(255,255,255,.12);
                color: #e6faff;
                font-size: .78rem;
            }}

            .readout-card strong {{
                display: block;
                color: var(--cyan);
                font-size: 1.05rem;
                margin-top: .2rem;
            }}

            .feature-grid, .metrics-grid, .module-grid {{
                display: grid;
                gap: 1rem;
            }}

            .feature-grid {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin: 1.1rem 0;
            }}

            .metrics-grid {{
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin: 1.1rem 0;
            }}

            .module-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}

            .feature-card, .metric-card, .module-card, .insight-panel {{
                padding: 1rem;
                transition: transform .24s ease, border-color .24s ease, box-shadow .24s ease;
            }}

            .feature-card:hover, .module-card:hover {{
                transform: translateY(-4px);
                border-color: rgba(46,233,255,.38);
                box-shadow: 0 24px 70px rgba(46, 233, 255, .16);
            }}

            .icon-tile {{
                width: 42px;
                height: 42px;
                border-radius: var(--radius);
                display: grid;
                place-items: center;
                font-weight: 800;
                color: #031321;
                background: linear-gradient(135deg, var(--cyan), var(--green));
                box-shadow: inset 0 1px 0 rgba(255,255,255,.34), 0 12px 26px rgba(46,233,255,.16);
                margin-bottom: .85rem;
            }}

            .feature-card h3, .module-card h3, .insight-panel h3 {{
                margin: 0 0 .4rem;
                font-size: 1rem;
                color: var(--text);
            }}

            .feature-card p, .module-card p, .metric-card p, .insight-panel p {{
                margin: 0;
                color: var(--muted);
                line-height: 1.55;
            }}

            .metric-card strong {{
                color: var(--text);
                display: block;
                font-size: clamp(1.65rem, 3vw, 2.4rem);
                line-height: 1;
                margin-bottom: .35rem;
            }}

            .section-title {{
                display: flex;
                align-items: end;
                justify-content: space-between;
                gap: 1rem;
                margin: 1.6rem 0 .85rem;
            }}

            .section-title h2 {{
                margin: 0;
                font-size: clamp(1.35rem, 3vw, 2rem);
            }}

            .section-title p {{
                margin: .3rem 0 0;
                color: var(--muted);
            }}

            .glass-pad {{
                padding: 1rem;
                margin-bottom: 1rem;
            }}

            .stButton > button, .stDownloadButton > button {{
                width: 100%;
                min-height: 42px;
                color: #04111f;
                background: linear-gradient(135deg, var(--cyan), var(--green));
                border: 0;
                border-radius: var(--radius);
                font-weight: 800;
                box-shadow: 0 16px 30px rgba(46, 233, 255, .18);
                transition: transform .22s ease, box-shadow .22s ease, filter .22s ease;
            }}

            .stButton > button:hover, .stDownloadButton > button:hover {{
                transform: translateY(-2px);
                filter: saturate(1.08);
                box-shadow: 0 20px 38px rgba(62, 229, 143, .20);
            }}

            .stButton > button[kind="secondary"] {{
                color: var(--text);
                border: 1px solid var(--border);
                background: var(--surface2);
                box-shadow: none;
            }}

            .stFileUploader [data-testid="stFileUploaderDropzone"] {{
                min-height: 150px;
                border: 1px dashed rgba(46,233,255,.42);
                border-radius: var(--radius);
                background: rgba(46,233,255,.07);
                transition: border-color .2s ease, background .2s ease, transform .2s ease;
            }}

            .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {{
                transform: translateY(-2px);
                border-color: rgba(62,229,143,.72);
                background: rgba(62,229,143,.08);
            }}

            .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {{
                border-radius: var(--radius);
                border-color: var(--border);
                background: rgba(255,255,255,.07);
                color: var(--text);
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: .4rem;
                background: var(--surface2);
                border: 1px solid var(--border);
                padding: .35rem;
                border-radius: var(--radius);
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: var(--radius);
                color: var(--muted);
                font-weight: 700;
            }}

            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, rgba(46,233,255,.18), rgba(138,92,255,.16));
                color: var(--text);
            }}

            div[data-testid="stMetric"] {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: .9rem;
                box-shadow: 0 16px 42px rgba(0,0,0,.18);
            }}

            div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
                color: var(--text);
            }}

            [data-testid="stDataFrame"], .stDataFrame {{
                border-radius: var(--radius);
                overflow: hidden;
            }}

            ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
            ::-webkit-scrollbar-track {{ background: rgba(255,255,255,.05); }}
            ::-webkit-scrollbar-thumb {{
                background: linear-gradient(180deg, var(--cyan), var(--purple));
                border-radius: 999px;
            }}

            .footer {{
                margin-top: 2rem;
                padding: 1rem;
                text-align: center;
            }}

            .footer a {{
                color: var(--cyan);
                text-decoration: none;
                font-weight: 700;
                margin: 0 .45rem;
            }}

            @media (max-width: 920px) {{
                .hero, .feature-grid, .metrics-grid, .module-grid {{
                    grid-template-columns: 1fr;
                }}

                .hero-copy, .ai-visual {{
                    min-height: auto;
                }}

                .ai-readout {{
                    grid-template-columns: 1fr;
                    position: relative;
                    margin-top: 290px;
                    left: auto;
                    right: auto;
                    bottom: auto;
                }}

                .scan-document {{
                    left: 8%;
                    right: 8%;
                    height: 250px;
                }}

                .section-title {{
                    display: block;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <div>
                <h2>{title}</h2>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-card">
                <p class="brand-title">{APP_NAME}</p>
                <p class="brand-subtitle">{APP_TAGLINE}</p>
                <div class="status-row">
                    <div class="status-pill"><span>OCR Engine</span><span class="pulse-dot"></span></div>
                    <div class="status-pill"><span>Risk Monitor</span><strong>Live</strong></div>
                    <div class="status-pill"><span>Workspace</span><strong>Secure</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        theme_is_light = st.toggle("Light mode", value=st.session_state.theme == "Light")
        st.session_state.theme = "Light" if theme_is_light else "Dark"

        pages = [
            "Command Center",
            "OCR Studio",
            "Multilingual OCR",
            "AI Assistant",
            "Document Classification",
            "Real-time OCR",
            "Financial Insights",
            "Loan Recommendation",
            "Fraud Detection",
            "PDF Analysis",
            "Profile & Access",
        ]
        active = st.radio("Navigation", pages, index=pages.index(st.session_state.active_page), label_visibility="collapsed")
        st.session_state.active_page = active

        st.markdown("### Notifications")
        for note in st.session_state.notifications[:3]:
            st.markdown(f'<div class="status-pill"><span>{note}</span><strong>OK</strong></div>', unsafe_allow_html=True)

    return st.session_state.active_page


def render_hero() -> None:
    st.markdown(
        """
        <div class="app-shell">
            <section class="hero">
                <div class="hero-copy glass">
                    <span class="eyebrow">AI fintech document intelligence</span>
                    <h1>FinOCR AI<br><span class="gradient-text">turns BFSI documents into decisions.</span></h1>
                    <p class="hero-lede">
                        Extract, classify, validate, and analyze bank statements, invoices, payslips,
                        tax forms, insurance records, and loan files with a secure OCR workflow built
                        for high-volume financial operations.
                    </p>
                    <div class="hero-actions">
                        <a class="hero-button primary" href="#ocr-studio">Launch OCR Studio</a>
                        <a class="hero-button secondary" href="#insights">View analytics</a>
                    </div>
                </div>
                <div class="ai-visual glass">
                    <div class="ai-frame">
                        <span class="neural-node node-1"></span>
                        <span class="neural-node node-2"></span>
                        <span class="neural-node node-3"></span>
                        <span class="neural-node node-4"></span>
                        <div class="scan-document">
                            <div class="doc-line long"></div>
                            <div class="doc-line medium"></div>
                            <div class="doc-line short"></div>
                            <div class="doc-line long"></div>
                            <div class="doc-line medium"></div>
                            <div class="doc-line short"></div>
                            <div class="scan-line"></div>
                        </div>
                        <div class="ai-readout">
                            <div class="readout-card">Confidence<strong>98.2%</strong></div>
                            <div class="readout-card">Latency<strong>0.8s</strong></div>
                            <div class="readout-card">Risk score<strong>Low</strong></div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_grid() -> None:
    features = [
        ("OCR", "Image and PDF extraction", "Drag and drop documents, preview images, tune preprocessing, and download clean text."),
        ("ML", "Document intelligence", "Auto-detect statements, invoices, payslips, tax records, insurance files, and loan documents."),
        ("AI", "Financial assistant", "Ask questions about extracted content, risk signals, eligibility, and operational next steps."),
        ("KPI", "Portfolio analytics", "Track document mix, processing confidence, cash-flow summaries, and anomaly trends."),
        ("RISK", "Fraud detection", "Score red flags such as low OCR confidence, suspicious wording, missing identifiers, and amount spikes."),
        ("LIVE", "Camera and voice workflows", "Capture webcam snapshots and voice notes for assisted field verification workflows."),
    ]
    html = ['<div class="feature-grid">']
    for icon, title, body in features:
        html.append(
            f"""
            <div class="feature-card">
                <div class="icon-tile">{icon}</div>
                <h3>{title}</h3>
                <p>{body}</p>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_stats() -> None:
    stats = [
        ("2.4M+", "pages processed in benchmark operations"),
        ("99.1%", "best-case printed statement OCR accuracy"),
        ("12", "supported BFSI document workflows"),
        ("24/7", "real-time operations console"),
    ]
    html = ['<div class="metrics-grid" id="insights">']
    for value, label in stats:
        html.append(f'<div class="metric-card"><strong>{value}</strong><p>{label}</p></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer glass">
            <strong>FinOCR AI</strong>
            <p>Secure OCR automation for banking, financial services, and insurance teams.</p>
            <a href="https://github.com" target="_blank">GitHub</a>
            <a href="https://www.linkedin.com" target="_blank">LinkedIn</a>
            <a href="https://x.com" target="_blank">X</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_to_png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def read_uploaded_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def preprocess_image(
    image: Image.Image,
    grayscale: bool = True,
    denoise: bool = False,
    threshold: bool = False,
    sharpen: bool = False,
    contrast: float = 1.15,
) -> Image.Image:
    processed = image.convert("RGB")

    if grayscale:
        processed = processed.convert("L")

    if contrast != 1:
        processed = ImageEnhance.Contrast(processed).enhance(contrast)

    if denoise:
        cv_image = np.array(processed)
        if cv_image.ndim == 2:
            cv_image = cv2.fastNlMeansDenoising(cv_image, None, 12, 7, 21)
        else:
            cv_image = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
        processed = Image.fromarray(cv_image)

    if threshold:
        gray = np.array(processed.convert("L"))
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        processed = Image.fromarray(thresh)

    if sharpen:
        processed = processed.filter(ImageFilter.SHARPEN)

    return processed


def classify_document(text: str) -> str:
    normalized = text.lower()
    scores = {
        doc_type: sum(1 for keyword in keywords if keyword in normalized)
        for doc_type, keywords in DOCUMENT_KEYWORDS.items()
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score else "Unclassified BFSI Document"


def safe_tesseract_call(image: Image.Image, lang: str, psm: int) -> OCRResult:
    config = f"--psm {psm}"
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DATAFRAME)
    except Exception as exc:
        return OCRResult(
            text=f"OCR engine error: {exc}",
            confidence=0.0,
            word_count=0,
            line_count=0,
            detected_type="OCR Engine Unavailable",
        )

    if data is None or data.empty or "conf" not in data:
        confidence = 0.0
    else:
        confidence_values = pd.to_numeric(data["conf"], errors="coerce")
        confidence_values = confidence_values[confidence_values >= 0]
        confidence = float(confidence_values.mean()) if not confidence_values.empty else 0.0

    words = re.findall(r"\b[\w.-]+\b", text)
    lines = [line for line in text.splitlines() if line.strip()]
    return OCRResult(
        text=text or "No readable text was detected. Try increasing contrast, enabling thresholding, or selecting another PSM mode.",
        confidence=round(confidence, 2),
        word_count=len(words),
        line_count=len(lines),
        detected_type=classify_document(text),
    )


def run_ocr_pipeline(
    image: Image.Image,
    lang: str,
    psm: int,
    grayscale: bool,
    denoise: bool,
    threshold: bool,
    sharpen: bool,
    contrast: float,
) -> tuple[Image.Image, OCRResult]:
    processed = preprocess_image(image, grayscale, denoise, threshold, sharpen, contrast)
    result = safe_tesseract_call(processed, lang, psm)
    st.session_state.latest_ocr_text = result.text
    return processed, result


def extract_amounts(text: str) -> list[float]:
    matches = re.findall(r"(?:rs\.?|inr|usd|eur|\$)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)", text, flags=re.I)
    amounts = []
    for match in matches:
        value = float(match.replace(",", ""))
        if value >= 10:
            amounts.append(value)
    return amounts[:80]


def fraud_rules(text: str, confidence: float) -> tuple[int, list[str]]:
    normalized = text.lower()
    flags = []

    if confidence and confidence < 55:
        flags.append("OCR confidence is low enough to require manual review.")
    if "urgent" in normalized or "immediately" in normalized:
        flags.append("Urgency language appears in the document text.")
    if "cash only" in normalized or "gift card" in normalized:
        flags.append("High-risk payment language was detected.")
    if "ifsc" not in normalized and "account" in normalized:
        flags.append("Banking content appears to be missing IFSC details.")
    if re.search(r"\b\d{9,18}\b", normalized) is None and any(word in normalized for word in ["bank", "account", "statement"]):
        flags.append("No clear account-length numeric identifier was found.")

    amounts = extract_amounts(text)
    if amounts:
        median_amount = float(np.median(amounts))
        max_amount = max(amounts)
        if median_amount and max_amount > median_amount * 8:
            flags.append("One transaction amount is much larger than the median amount.")
        if sum(1 for amount in amounts if amount % 10000 == 0) >= 3:
            flags.append("Several round-number transactions were found.")

    risk_score = min(100, 18 + len(flags) * 17 + (10 if confidence < 65 else 0))
    return risk_score, flags or ["No major heuristic risk flags were detected."]


def render_copy_button(text: str, height: int = 72) -> None:
    payload = json.dumps(text)
    components.html(
        f"""
        <div style="font-family: Inter, system-ui; display:flex; gap:10px; align-items:center;">
            <button
                onclick='navigator.clipboard.writeText({payload}); this.innerText="Copied";'
                style="
                    min-height:40px;
                    border:0;
                    border-radius:8px;
                    padding:0 16px;
                    color:#04111f;
                    font-weight:800;
                    background:linear-gradient(135deg,#2ee9ff,#3ee58f);
                    cursor:pointer;
                ">
                Copy text
            </button>
            <span style="color:#94a3b8;font-size:13px;">Copies extracted text to clipboard.</span>
        </div>
        """,
        height=height,
    )


def render_ocr_result(result: OCRResult) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Confidence", f"{result.confidence:.1f}%")
    metric_cols[1].metric("Words", f"{result.word_count:,}")
    metric_cols[2].metric("Lines", f"{result.line_count:,}")
    metric_cols[3].metric("Detected type", result.detected_type)

    st.text_area("Extracted text", value=result.text, height=260)
    actions = st.columns([1, 1])
    with actions[0]:
        st.download_button(
            "Download extracted text",
            result.text.encode("utf-8"),
            file_name=f"finocr-extraction-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
            mime="text/plain",
        )
    with actions[1]:
        render_copy_button(result.text)


def render_ocr_controls(prefix: str = "") -> dict[str, object]:
    with st.expander("OCR preprocessing options", expanded=True):
        cols = st.columns(4)
        grayscale = cols[0].toggle("Grayscale", value=True, key=f"{prefix}gray")
        denoise = cols[1].toggle("Denoise", value=False, key=f"{prefix}denoise")
        threshold = cols[2].toggle("Adaptive threshold", value=False, key=f"{prefix}threshold")
        sharpen = cols[3].toggle("Sharpen", value=True, key=f"{prefix}sharpen")

        lang_col, psm_col, contrast_col = st.columns([1.4, 1, 1])
        language_label = lang_col.selectbox("Language", list(LANGUAGE_OPTIONS.keys()), key=f"{prefix}lang")
        psm = psm_col.selectbox("Page segmentation", [3, 4, 6, 11, 12], index=2, key=f"{prefix}psm")
        contrast = contrast_col.slider("Contrast", 0.75, 2.0, 1.15, 0.05, key=f"{prefix}contrast")

    return {
        "lang": LANGUAGE_OPTIONS[language_label],
        "psm": psm,
        "grayscale": grayscale,
        "denoise": denoise,
        "threshold": threshold,
        "sharpen": sharpen,
        "contrast": contrast,
    }


def command_center() -> None:
    render_hero()
    section_title("Platform Modules", "A premium command center for OCR, analytics, risk, and financial decision support.")
    render_feature_grid()
    section_title("Operating Snapshot", "Animated platform metrics and high-level operational signals.")
    render_stats()

    st.markdown('<div id="ocr-studio"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, 1])
    with left:
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=SAMPLE_TRANSACTIONS["Month"], y=SAMPLE_TRANSACTIONS["Income"], name="Income", mode="lines+markers"))
        chart.add_trace(go.Scatter(x=SAMPLE_TRANSACTIONS["Month"], y=SAMPLE_TRANSACTIONS["Expenses"], name="Expenses", mode="lines+markers"))
        chart.update_layout(
            title="Cash-flow trend",
            template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white",
            margin=dict(l=20, r=20, t=55, b=20),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.markdown(
            """
            <div class="insight-panel">
                <h3>AI Review Queue</h3>
                <p>18 documents are ready for validation. Five need a second pass because of low contrast, missing IDs, or unusual transaction amounts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Automation rate", "86%", "+8.4%")
        st.metric("Avg review time", "41 sec", "-18 sec")
    render_footer()


def ocr_studio() -> None:
    section_title("OCR Studio", "Upload images, preview preprocessing, extract text, score confidence, copy results, and export reports.")
    uploaded = st.file_uploader("Drag and drop a bank statement, invoice, payslip, cheque, or tax image", type=["png", "jpg", "jpeg"])
    options = render_ocr_controls("studio_")

    if uploaded:
        original = read_uploaded_image(uploaded)
        with st.spinner("Scanning document with FinOCR AI..."):
            processed, result = run_ocr_pipeline(original, **options)

        preview_col, result_col = st.columns([1, 1.05])
        with preview_col:
            st.image(original, caption="Original upload", use_container_width=True)
            st.image(processed, caption="Preprocessed preview", use_container_width=True)
        with result_col:
            render_ocr_result(result)
    else:
        st.info("Upload a PNG or JPG to start extraction. The drop zone supports drag and drop.")


def multilingual_ocr() -> None:
    section_title("Multilingual OCR", "Extract multilingual financial documents with Tesseract language packs.")
    language = st.selectbox("Select OCR language model", list(LANGUAGE_OPTIONS.keys()), index=0)
    uploaded = st.file_uploader("Upload multilingual image", type=["png", "jpg", "jpeg"])
    cols = st.columns(3)
    denoise = cols[0].toggle("Denoise", value=True)
    threshold = cols[1].toggle("Threshold", value=False)
    sharpen = cols[2].toggle("Sharpen", value=True)

    if uploaded:
        image = read_uploaded_image(uploaded)
        with st.spinner(f"Running {language} OCR..."):
            processed, result = run_ocr_pipeline(
                image,
                lang=LANGUAGE_OPTIONS[language],
                psm=6,
                grayscale=True,
                denoise=denoise,
                threshold=threshold,
                sharpen=sharpen,
                contrast=1.2,
            )
        left, right = st.columns(2)
        left.image(processed, caption="Optimized multilingual input", use_container_width=True)
        with right:
            render_ocr_result(result)
    else:
        st.info("Install the matching Tesseract language data files for best multilingual extraction.")


def assistant_reply(prompt: str) -> str:
    text = st.session_state.latest_ocr_text
    normalized = prompt.lower()

    if "summary" in normalized or "summarize" in normalized:
        if not text:
            return "Upload and extract a document first, then I can summarize the financial content."
        detected = classify_document(text)
        amounts = extract_amounts(text)
        amount_line = f"I found {len(amounts)} numeric monetary values; the largest visible value is {max(amounts):,.2f}." if amounts else "I did not find clear monetary values."
        return f"This appears to be a {detected}. {amount_line} Review identifiers, dates, and totals before downstream use."

    if "fraud" in normalized or "risk" in normalized:
        score, flags = fraud_rules(text or prompt, 72 if text else 50)
        return f"Heuristic risk score: {score}/100. Key signal: {flags[0]}"

    if "loan" in normalized:
        return "For loans, I would compare CIBIL score, income stability, debt-to-income ratio, course or asset value, collateral, and repayment tenure. Use the Loan Recommendation module for a scored table."

    if "ocr" in normalized or "extract" in normalized:
        return "Use OCR Studio for single documents, Multilingual OCR for language packs, PDF Analysis for page extraction, and Real-time OCR for webcam snapshots."

    return "I can help with document summaries, fraud risk, loan eligibility, OCR setup, and financial insights. Upload a document first for document-aware answers."


def ai_assistant() -> None:
    section_title("AI Financial Assistant", "A document-aware assistant for OCR summaries, financial questions, fraud signals, and workflow guidance.")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask about OCR, a document summary, fraud risk, or loan options")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        reply = assistant_reply(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


def process_upload_for_text(uploaded_file, lang: str = "eng") -> OCRResult:
    image = read_uploaded_image(uploaded_file)
    processed = preprocess_image(image, grayscale=True, denoise=True, threshold=False, sharpen=True, contrast=1.2)
    return safe_tesseract_call(processed, lang=lang, psm=6)


def document_classification() -> None:
    section_title("Document Classification", "Batch classify BFSI files and visualize document mix, OCR confidence, and text density.")
    uploads = st.file_uploader("Upload multiple document images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if not uploads:
        st.info("Upload up to 10 images to build a classification report.")
        return

    rows = []
    progress = st.progress(0)
    for index, file in enumerate(uploads[:10], start=1):
        result = process_upload_for_text(file)
        rows.append(
            {
                "File": file.name,
                "Detected Type": result.detected_type,
                "Confidence": result.confidence,
                "Words": result.word_count,
                "Lines": result.line_count,
            }
        )
        progress.progress(index / min(len(uploads), 10))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        pie = px.pie(df, names="Detected Type", title="Document mix", hole=0.45)
        pie.update_layout(template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white")
        st.plotly_chart(pie, use_container_width=True)
    with col2:
        bar = px.bar(df, x="File", y="Confidence", color="Detected Type", title="OCR confidence by file")
        bar.update_layout(template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white")
        st.plotly_chart(bar, use_container_width=True)

    st.download_button(
        "Export classification report",
        df.to_csv(index=False).encode("utf-8"),
        "finocr-classification-report.csv",
        "text/csv",
    )


def real_time_ocr() -> None:
    section_title("Real-time OCR", "Capture webcam snapshots and voice notes for assisted field verification.")
    camera_tab, voice_tab = st.tabs(["Webcam OCR", "Voice OCR"])

    with camera_tab:
        st.caption("Use the camera input to capture a document snapshot, then run OCR on the frame.")
        camera_image = st.camera_input("Capture document frame")
        options = render_ocr_controls("camera_")
        if camera_image:
            image = read_uploaded_image(camera_image)
            with st.spinner("Reading webcam frame..."):
                processed, result = run_ocr_pipeline(image, **options)
            left, right = st.columns(2)
            left.image(processed, caption="Processed camera frame", use_container_width=True)
            with right:
                render_ocr_result(result)

    with voice_tab:
        st.caption("Capture an analyst note and attach it to the current OCR workflow.")
        audio_file = None
        if hasattr(st, "audio_input"):
            audio_file = st.audio_input("Record a voice note")
        else:
            audio_file = st.file_uploader("Upload WAV audio note", type=["wav", "mp3", "m4a"])

        if audio_file:
            st.audio(audio_file)
            transcript = transcribe_audio(audio_file)
            st.text_area("Voice note transcript", transcript, height=180)
            st.download_button("Export voice note transcript", transcript.encode("utf-8"), "finocr-voice-note.txt", "text/plain")


def transcribe_audio(audio_file) -> str:
    try:
        import speech_recognition as sr
    except ImportError:
        return "Voice capture is available. Install SpeechRecognition and an offline recognizer such as pocketsphinx to enable transcription."

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_sphinx(audio)
    except Exception as exc:
        return f"Voice transcription could not be completed locally: {exc}"


def financial_insights() -> None:
    section_title("Financial Insights Dashboard", "Interactive KPI cards, financial analytics, document insights, and real-time visualization.")

    source_text = st.text_area("Optional extracted OCR text for amount analytics", st.session_state.latest_ocr_text, height=160)
    amounts = extract_amounts(source_text)

    if amounts:
        amount_df = pd.DataFrame({"Index": list(range(1, len(amounts) + 1)), "Amount": amounts})
        total_value = sum(amounts)
        avg_value = float(np.mean(amounts))
        max_value = max(amounts)
    else:
        amount_df = SAMPLE_TRANSACTIONS.melt(id_vars="Month", value_vars=["Income", "Expenses"], var_name="Type", value_name="Amount")
        total_value = int(SAMPLE_TRANSACTIONS["Income"].sum())
        avg_value = int(SAMPLE_TRANSACTIONS["Expenses"].mean())
        max_value = int(SAMPLE_TRANSACTIONS["Income"].max())

    cols = st.columns(4)
    cols[0].metric("Total detected value", f"{total_value:,.0f}", "+12.4%")
    cols[1].metric("Average value", f"{avg_value:,.0f}", "+4.2%")
    cols[2].metric("Largest value", f"{max_value:,.0f}", "Review")
    cols[3].metric("Documents analyzed", "128", "+19")

    chart_col, pie_col = st.columns([1.25, 1])
    template = "plotly_dark" if st.session_state.theme == "Dark" else "plotly_white"
    with chart_col:
        if amounts:
            line = px.line(amount_df, x="Index", y="Amount", markers=True, title="Extracted monetary values")
        else:
            line = px.bar(amount_df, x="Month", y="Amount", color="Type", barmode="group", title="Income versus expenses")
        line.update_layout(template=template, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(line, use_container_width=True)
    with pie_col:
        mix = pd.DataFrame({"Type": ["Statements", "Invoices", "Payslips", "Loans", "Insurance"], "Count": [42, 31, 18, 21, 16]})
        pie = px.pie(mix, names="Type", values="Count", hole=0.52, title="Document insights")
        pie.update_layout(template=template, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(pie, use_container_width=True)

    risk = px.area(SAMPLE_TRANSACTIONS, x="Month", y="Risk Events", title="Risk event trend")
    risk.update_layout(template=template, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(risk, use_container_width=True)


def loan_recommendation() -> None:
    section_title("AI Loan Recommendation System", "Score borrower fit and compare recommended BFSI loan products.")
    input_col, output_col = st.columns([1, 1.2])
    with input_col:
        applicant_type = st.selectbox("Applicant segment", ["Student", "Salaried", "Self-employed", "MSME"])
        monthly_income = st.number_input("Monthly income", min_value=0, value=85000, step=5000)
        cibil = st.slider("CIBIL score", 300, 900, 742)
        obligation = st.number_input("Existing monthly obligations", min_value=0, value=18000, step=1000)
        requested_amount = st.number_input("Requested loan amount", min_value=10000, value=1200000, step=50000)
        tenure = st.slider("Preferred tenure in years", 1, 20, 7)

    disposable = max(monthly_income - obligation, 0)
    dti = obligation / monthly_income if monthly_income else 1
    eligibility = min(100, max(0, (cibil - 300) / 6 + disposable / 3500 - dti * 25))
    risk_band = "Prime" if eligibility >= 78 else "Near-prime" if eligibility >= 58 else "Manual review"

    with output_col:
        st.metric("Eligibility score", f"{eligibility:.0f}/100", risk_band)
        st.metric("Debt-to-income", f"{dti * 100:.1f}%")
        st.metric("Indicative EMI capacity", f"{disposable * 0.45:,.0f}")

        products = pd.DataFrame(
            [
                {"Product": "FinOCR Smart Personal Loan", "APR": "10.4%-13.9%", "Fit": min(100, eligibility + 6), "Max Amount": requested_amount * 1.15},
                {"Product": "Education Growth Loan", "APR": "8.2%-11.5%", "Fit": min(100, eligibility + (10 if applicant_type == "Student" else 0)), "Max Amount": requested_amount},
                {"Product": "Secured Asset Loan", "APR": "9.1%-12.2%", "Fit": min(100, eligibility + 12), "Max Amount": requested_amount * 1.35},
                {"Product": "MSME Working Capital", "APR": "11.8%-15.5%", "Fit": min(100, eligibility + (12 if applicant_type == "MSME" else -5)), "Max Amount": requested_amount * 1.1},
            ]
        ).sort_values("Fit", ascending=False)
        st.dataframe(products, use_container_width=True)

    fig = px.bar(products, x="Product", y="Fit", color="Fit", title="Recommendation fit score", range_y=[0, 100])
    fig.update_layout(template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def fraud_detection() -> None:
    section_title("Fraud Detection", "Upload a document or use the latest OCR text to score risk signals.")
    uploaded = st.file_uploader("Upload document image for fraud scan", type=["png", "jpg", "jpeg"])
    text = st.session_state.latest_ocr_text
    confidence = 72.0

    if uploaded:
        result = process_upload_for_text(uploaded)
        text = result.text
        confidence = result.confidence
        st.session_state.latest_ocr_text = text
        st.success(f"Document scanned as {result.detected_type}.")
    else:
        text = st.text_area("Paste text for fraud analysis", value=text, height=210)

    risk_score, flags = fraud_rules(text, confidence)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Heuristic risk score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff6b7a" if risk_score >= 70 else "#f7c948" if risk_score >= 45 else "#3ee58f"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(62,229,143,.18)"},
                    {"range": [40, 70], "color": "rgba(247,201,72,.20)"},
                    {"range": [70, 100], "color": "rgba(255,107,122,.22)"},
                ],
            },
        )
    )
    gauge.update_layout(template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white", height=320)

    left, right = st.columns([1, 1])
    left.plotly_chart(gauge, use_container_width=True)
    with right:
        st.markdown('<div class="insight-panel"><h3>Detected risk signals</h3></div>', unsafe_allow_html=True)
        for flag in flags:
            st.warning(flag)


def pdf_pages_from_bytes(pdf_bytes: bytes, max_pages: int) -> list[Image.Image]:
    if fitz is None:
        return []

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in document[:max_pages]:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pages.append(Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB"))
    return pages


def pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages[:8]:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def pdf_analysis() -> None:
    section_title("PDF Upload & Analysis", "Render PDF pages, run OCR, classify documents, and export an analysis report.")
    uploaded = st.file_uploader("Upload PDF document", type=["pdf"])
    page_limit = st.slider("Pages to process", 1, 8, 3)
    options = render_ocr_controls("pdf_")

    if not uploaded:
        st.info("Upload a PDF. PyMuPDF enables OCR rendering; pypdf enables embedded text extraction.")
        return

    pdf_bytes = uploaded.getvalue()
    embedded_text = pdf_text_from_bytes(pdf_bytes)
    rendered_pages = pdf_pages_from_bytes(pdf_bytes, page_limit)

    ocr_texts = []
    page_rows = []
    if rendered_pages:
        progress = st.progress(0)
        for idx, page_image in enumerate(rendered_pages, start=1):
            _, result = run_ocr_pipeline(page_image, **options)
            ocr_texts.append(result.text)
            page_rows.append({"Page": idx, "Type": result.detected_type, "Confidence": result.confidence, "Words": result.word_count})
            progress.progress(idx / len(rendered_pages))
    elif not embedded_text:
        st.error("PDF analysis needs PyMuPDF or pypdf installed. Add pymupdf or pypdf to your environment and rerun.")
        return

    combined_text = "\n\n".join([embedded_text, *ocr_texts]).strip()
    st.session_state.latest_ocr_text = combined_text

    if page_rows:
        st.dataframe(pd.DataFrame(page_rows), use_container_width=True)

    st.text_area("Combined PDF text", combined_text or "No text extracted.", height=300)
    st.download_button(
        "Export PDF analysis text",
        (combined_text or "").encode("utf-8"),
        "finocr-pdf-analysis.txt",
        "text/plain",
    )


def profile_access() -> None:
    section_title("Login, Signup, and Profile", "A premium access-management UI for secure BFSI workflows.")
    login_tab, signup_tab, profile_tab = st.tabs(["Login", "Signup", "Profile"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Work email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            st.session_state.signed_in = bool(email and password)
            if st.session_state.signed_in:
                st.success("Signed in to the FinOCR AI workspace.")
            else:
                st.error("Enter email and password to continue.")

    with signup_tab:
        with st.form("signup_form"):
            name = st.text_input("Full name")
            organization = st.text_input("Organization")
            role = st.selectbox("Role", ["Credit Operations", "Risk Analyst", "Loan Officer", "Insurance Ops", "Administrator"])
            submitted = st.form_submit_button("Create workspace")
        if submitted:
            st.session_state.profile = {"name": name or "BFSI User", "role": role, "workspace": organization or "FinOCR AI"}
            st.session_state.signed_in = True
            st.success("Workspace profile created.")

    with profile_tab:
        profile = st.session_state.profile
        st.markdown(
            f"""
            <div class="insight-panel">
                <h3>{profile['name']}</h3>
                <p>{profile['role']} at {profile['workspace']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        cols[0].metric("Documents", "1,248")
        cols[1].metric("Accuracy", "96.8%")
        cols[2].metric("Risk alerts", "27")
        st.toggle("Email notifications", value=True)
        st.toggle("Require manual review for low confidence OCR", value=True)
        st.toggle("Dark mode follows system", value=False)


def render_page(page: str) -> None:
    pages = {
        "Command Center": command_center,
        "OCR Studio": ocr_studio,
        "Multilingual OCR": multilingual_ocr,
        "AI Assistant": ai_assistant,
        "Document Classification": document_classification,
        "Real-time OCR": real_time_ocr,
        "Financial Insights": financial_insights,
        "Loan Recommendation": loan_recommendation,
        "Fraud Detection": fraud_detection,
        "PDF Analysis": pdf_analysis,
        "Profile & Access": profile_access,
    }
    pages[page]()


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="FIN",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    configure_tesseract()
    init_state()
    inject_css(st.session_state.theme)
    active_page = render_sidebar()
    render_page(active_page)


if __name__ == "__main__":
    main()