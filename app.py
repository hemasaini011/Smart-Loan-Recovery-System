import time
import random
import string
import math
import re
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import streamlit.components.v1 as components

# -----------------------------------------------
# Load Model and Features
# -----------------------------------------------
with open("models/kmeans.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/features.pkl", "rb") as f:
    features_list = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/segment_names.pkl", "rb") as f:
    segment_names = pickle.load(f)

# -----------------------------------------------
# Utility Functions
# -----------------------------------------------

def generate_borrower_id(loan_type, first_name, last_name):
    """Generate a unique borrower ID for PDF."""
    loan_code = loan_type[:3].upper() if loan_type else "GEN"
    initials = (first_name[0].upper() if first_name else "X") + (last_name[0].upper() if last_name else "X")
    timestamp = str(int(time.time()))
    random_hex = "".join(random.choices(string.hexdigits.upper(), k=4))
    return f"{loan_code}-{initials}-{timestamp}-{random_hex}"


def get_default_loan_terms(loan_type):
    """Return typical interest rate and tenure for a given loan type."""
    loan_type = loan_type.lower()
    mapping = {
        "personal": (14.0, 48),
        "auto": (10.5, 60),
        "business": (16.0, 36),
        "home": (9.0, 180),
    }
    return mapping.get(loan_type, (15.0, 36))


def assign_recovery_strategy(score, dpd):
    """Return a recovery strategy and risk category based on risk score and DPD."""
    if score > 0.90:
        if dpd >= 90:
            return (
                "Initiate legal proceedings, send final demand notices with collateral seizure intent, "
                "escalate case to external recovery agencies, and flag borrower as a chronic defaulter.",
                "Critical Risk",
            )
        else:
            return (
                "Send pre-litigation warning, offer limited time restructuring, and escalate to senior recovery team.",
                "High Risk",
            )
    elif 0.75 < score <= 0.90:
        return (
            "Offer one-time settlement options or revised repayment terms, "
            "escalate to senior collections team, and issue a pre-litigation warning.",
            "High Risk",
        )
    elif 0.25 < score <= 0.75:
        return (
            "Trigger multiple soft recovery attempts including calls, emails, and WhatsApp nudges. "
            "Offer flexible EMI restructuring plans and conduct borrower behavior analysis.",
            "Medium Risk",
        )
    else:
        return (
            "Send timely automated reminders via SMS/email, monitor payment behavior closely, "
            "and provide financial advisory nudges to maintain repayment consistency.",
            "Low Risk",
        )


def calculate_emi(P, annual_rate, n):
    """Calculate monthly EMI given principal, annual rate, and tenure in months."""
    if P in (None, 0) or annual_rate in (None, 0) or n in (None, 0):
        return None
    r = annual_rate / (12 * 100)
    emi = (P * r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return round(emi, 2)


def remove_emoji(text):
    """Remove non-ASCII characters (emojis) for PDF compatibility."""
    return re.sub(r"[^\x00-\x7F]+", "", str(text))


def is_missing(val):
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def is_invalid(val):
    return val is None or (isinstance(val, float) and math.isnan(val))


# -----------------------------------------------
# Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Smart Loan Recovery System",
    page_icon="🏦",
    layout="wide",
)

# -----------------------------------------------
# Global CSS
# -----------------------------------------------
st.markdown(
    """
    <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1A2340;
            padding-top: 20px;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            background-color: #1B3A6B;
            color: #FFFFFF;
            padding: 10px 24px;
            border: 1.5px solid #4A90D9;
            font-weight: 600;
            transition: 0.25s ease;
        }
        .stButton > button:hover {
            background-color: #4A90D9;
            color: #FFFFFF;
            transform: scale(1.03);
        }

        /* Headings */
        h1, h2, h3 {
            color: #E8F0FE;
        }

        /* Sidebar header card */
        .sidebar-header {
            background: #0F1F3D;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 18px;
            color: #E8F0FE;
            font-size: 17px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            border: 1.5px solid #2E4A7A;
        }
        .sidebar-header img {
            width: 46px;
            height: 46px;
        }

        /* Info/result cards */
        .info-card {
            background: rgba(15, 31, 61, 0.90);
            border-radius: 14px;
            padding: 20px;
            margin-top: 14px;
            color: #E8F0FE;
            font-size: 1.05rem;
            line-height: 1.75;
            border: 1.5px solid #2E4A7A;
        }

        /* ------------------- INPUT STYLING ------------------- */

        /* Labels */
        label {
            color: #FFFFFF !important;
        }

        /* Input fields */
        input, textarea {
            color: #4A90D9 !important;   /* 🔵 BLUE TEXT */
            background-color: rgba(255,255,255,0.08) !important;
            border: 1px solid #4A90D9 !important;
            border-radius: 6px !important;
        }

        /* Placeholder */
        input::placeholder {
            color: #CCCCCC !important;
        }

        /* Selectbox text */
        div[data-baseweb="select"] span {
            color: #4A90D9 !important;
        }

        /* Number input */
        input[type="number"] {
            color: #4A90D9 !important;
        }

        /* Focus effect */
        input:focus, textarea:focus {
            outline: none !important;
            border: 1.5px solid #4A90D9 !important;
            box-shadow: 0 0 8px #4A90D9 !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------
# Sidebar
# -----------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <img src="https://cdn-icons-png.flaticon.com/512/2830/2830284.png" />
            <span>Smart Loan<br>Recovery System</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = option_menu(
        menu_title="Navigation",
        options=["Overview", "Smart Risk Predictor", "Recovery Insights", "Dashboard"],
        icons=["house-door", "graph-up-arrow", "bar-chart-line", "speedometer2"],
        menu_icon="bank2",
        default_index=0,
        styles={
            "container": {
                "padding": "6px",
                "background-color": "#1A2340"
            },

            "icon": {
                "color": "#4A90D9",
                "font-size": "18px"
            },

            "menu-title": {
                "color": "#FFD700",
                "font-size": "18px",
                "font-weight": "700"
            },

            # Normal menu item
            "nav-link": {
                "color": "#C5D3E8",
                "font-size": "15px",
                "text-align": "left",
                "margin": "3px",
                "--hover-color": "#FFD700",   # 🔥 hover yellow
                "border-radius": "8px"
            },

            # 🔥 ACTIVE ITEM (YELLOW)
            "nav-link-selected": {
                "background-color": "#FFD700",   # yellow background
                "color": "#000000",              # black text
                "font-weight": "700",
                "border-radius": "8px",
                "box-shadow": "0 0 10px rgba(255, 215, 0, 0.6)"  # glow effect
            },
        },
    )

    st.markdown(
        """
        <hr style="border-color:#2E4A7A; margin-top:28px; margin-bottom:10px;">
        <p style="text-align:center; color:#7A8FAF; font-size:13px; margin:0;">
            Loan Analytics System
        </p>
        """,
        unsafe_allow_html=True,
    )
# ================================================
# PAGE 1: OVERVIEW
# ================================================
if page == "Overview":
    # Background
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://images.unsplash.com/photo-1563986768494-4dee2763ff3f?w=1800&auto=format&fit=crop&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background: rgba(8, 20, 45, 0.72);
            z-index: 0;
        }
        [data-testid="stAppViewContainer"] > * {
            position: relative;
            z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
    """
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <img src="https://cdn-icons-png.flaticon.com/512/2830/2830284.png"
             width="72" style="margin-bottom:12px;" />
        <h1 style="font-size:2.8em; color:#E8F0FE; font-weight:800; margin:0;">
            Smart Loan Recovery System
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

    # About card
    st.markdown(
        """
        <div class="info-card" style="margin-top:24px;">
            This platform uses <b>machine learning</b> to predict the risk of borrower default
            and suggests <b>optimal recovery strategies</b> for financial institutions.<br><br>
            <b>✔ Borrower Segmentation</b> using K-Means Clustering into 4 meaningful groups.<br>
            <b>✔ Risk Prediction</b> with a Random Forest model trained on real loan data.<br>
            <b>✔ Actionable Strategies</b> tailored to each borrower's risk profile.<br>
            <b>✔ Visual Analytics</b> — charts and insights for recovery teams.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature highlights
    col1, col2, col3 = st.columns(3)
    cards = [
        (
            "https://cdn-icons-png.flaticon.com/512/2920/2920244.png",
            "Predict Risk Instantly",
            "Enter borrower details to get an AI-powered risk score and recovery recommendation in seconds.",
        ),
        (
            "https://cdn-icons-png.flaticon.com/512/1584/1584942.png",
            "Visualize Segments",
            "See borrower clusters, risk categories, and key financial ratios in an easy-to-read dashboard.",
        ),
        (
            "https://cdn-icons-png.flaticon.com/512/3135/3135706.png",
            "Download PDF Reports",
            "Generate professional borrower risk reports ready for review and record-keeping.",
        ),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f"""
                <div class="info-card" style="text-align:center; min-height:200px;">
                    <img src="{icon}" width="48" style="margin-bottom:10px;" /><br>
                    <b style="font-size:1.1em; color:#4A90D9;">{title}</b><br>
                    <span style="font-size:0.97em; color:#C5D3E8;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # How to use
    st.markdown(
        """
        <div class="info-card" style="margin-top:24px;">
            <b style="color:#4A90D9; font-size:1.08em;">How to Use This Application</b><br><br>
            1. Go to <b>Smart Risk Predictor</b> from the sidebar menu.<br>
            2. Enter the borrower's loan and financial details.<br>
            3. Click <b>Predict Risk &amp; Strategy</b> to get the result.<br>
            4. View detailed analytics under <b>Recovery Insights</b>.<br>
            5. Download a professional <b>PDF report</b> for the borrower.<br>
            6. Explore the <b>Dashboard</b> for a comprehensive overview of loan recovery analytics.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ================================================
# PAGE 2: SMART RISK PREDICTOR
# ================================================
elif page == "Smart Risk Predictor":
    # Background
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://images.unsplash.com/photo-1728267900779-b429d7c3194f?q=80&w=1800&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
        }
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(8, 20, 45, 0.65);
            z-index: 0;
            pointer-events: none;
        }
        [data-testid="stVerticalBlock"] {
            position: relative;
            z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Page title
    st.markdown(
        """
        <h1 style="text-align:center; color:#E8F0FE; font-size:2.4em; font-weight:800; margin-bottom:4px;">
            Smart Risk Predictor
        </h1>
        <p style="text-align:center; color:#A8C0E0; font-size:1.05em; margin-bottom:20px;">
            Enter borrower details below to predict default risk and receive a recovery strategy.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Form container
    st.markdown(
        """
        <div style="background:rgba(10,22,46,0.88); border-radius:16px; padding:22px 20px;
                    border:1.5px solid #2E4A7A; margin-bottom:18px;">
        """,
        unsafe_allow_html=True,
    )

    # Loan type selection
    loan_type_options = ["Select Loan Type", "Personal", "Auto", "Business", "Home"]
    loan_type = st.selectbox("Select Loan Type", loan_type_options, index=0, key="loan_type_select")

    if loan_type == "Select Loan Type":
        st.info("Please select a loan type to proceed.")
        default_interest, default_tenure = None, None
        custom_scheme = False
    else:
        default_interest, default_tenure = get_default_loan_terms(loan_type)
        st.markdown(
            f"<p style='color:#A8C0E0;'>Typical terms for a <b>{loan_type}</b> loan: "
            f"<b>{default_interest}% interest</b> for <b>{default_tenure} months</b></p>",
            unsafe_allow_html=True,
        )
        custom_scheme = st.checkbox("Loan applied during a special scheme or offer?")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        first_name = st.text_input("First Name", placeholder="e.g. Rahul")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=18, max_value=100, value=None, placeholder="e.g. 35")
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0.0, value=None, placeholder="e.g. 50000")
        num_dependents = st.number_input("Number of Dependents", min_value=0, step=1, value=None, placeholder="e.g. 2")
        loan_amount = st.number_input("Loan Amount (₹)", min_value=10000.0, value=None, placeholder="e.g. 300000")
        collateral_value = st.number_input("Collateral Value (₹)", min_value=0.0, value=None, placeholder="e.g. 200000")

    with col2:
        last_name = st.text_input("Last Name", placeholder="e.g. Sharma")

        if loan_type != "Select Loan Type":
            if custom_scheme:
                interest_rate = st.number_input(
                    "Custom Interest Rate (%)", min_value=0.0, max_value=100.0,
                    value=default_interest, step=0.1, format="%.2f"
                )
                loan_tenure = st.number_input(
                    "Custom Tenure (Months)", min_value=1, max_value=360,
                    value=default_tenure, step=1
                )
            else:
                interest_rate = default_interest
                loan_tenure = default_tenure
                st.number_input("Interest Rate (%)", value=interest_rate, disabled=True,
                                help="Default rate for selected loan type.")
                st.number_input("Loan Tenure (Months)", value=loan_tenure, disabled=True,
                                help="Default tenure for selected loan type.")
        else:
            interest_rate = None
            loan_tenure = None
            st.number_input("Interest Rate (%)", value=None, disabled=True)
            st.number_input("Loan Tenure (Months)", value=None, disabled=True)

        outstanding_loan = st.number_input("Outstanding Loan Amount (₹)", min_value=0.0, value=None, placeholder="e.g. 150000")
        missed_payments = st.number_input("Missed Payments", min_value=0, step=1, value=None, placeholder="e.g. 1")

        # Auto-calculated fields
        days_past_due = (missed_payments or 0) * 30
        st.number_input("Days Past Due (auto-calculated)", value=days_past_due if missed_payments is not None else None,
                        disabled=True, help="Calculated as: missed payments × 30 days.")

        if missed_payments == 0:
            collection_attempts = 0
        elif days_past_due <= 30:
            collection_attempts = 1
        elif days_past_due <= 60:
            collection_attempts = 2
        elif days_past_due <= 90:
            collection_attempts = 3
        else:
            collection_attempts = 4
        st.number_input("Collection Attempts (auto)", value=collection_attempts if missed_payments is not None else None,
                        disabled=True, help="Based on Days Past Due.")

    # EMI display
    monthly_emi = calculate_emi(loan_amount, interest_rate, loan_tenure)
    if monthly_emi is not None:
        st.markdown(
            f"<p style='color:#E8F0FE; font-size:1.1em; margin-top:10px;'>"
            f"<b>Calculated Monthly EMI:</b> ₹{monthly_emi:,.2f}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='color:#A8C0E0; font-size:1.1em;'><b>Monthly EMI:</b> —</p>",
            unsafe_allow_html=True,
        )

    # Derived features
    emi_to_income = (
        round(monthly_emi / monthly_income, 3)
        if (monthly_income not in (None, 0, 0.0) and monthly_emi not in (None, 0, 0.0))
        else None
    )
    collateral_coverage = (
        round(collateral_value / loan_amount, 3)
        if (loan_amount not in (None, 0, 0.0) and collateral_value is not None)
        else None
    )
    default_severity = (
        missed_payments * days_past_due
        if (missed_payments is not None and days_past_due is not None)
        else None
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("Predict Risk & Strategy", use_container_width=True)

    if predict_clicked:
        any_missing = (
            is_missing(first_name) or is_missing(last_name) or is_missing(age)
            or is_missing(monthly_income) or is_missing(num_dependents)
            or is_missing(loan_amount) or is_missing(loan_tenure) or is_missing(interest_rate)
            or is_missing(outstanding_loan) or is_missing(missed_payments) or is_missing(collateral_value)
            or loan_type == "Select Loan Type"
        )

        if any_missing or is_invalid(emi_to_income) or is_invalid(collateral_coverage) or is_invalid(default_severity):
            st.error("Please fill in all fields correctly before predicting.")
        elif monthly_income == 0:
            st.error("Monthly income cannot be zero.")
        else:
            # Build input and run prediction
            input_array = np.array([[
                age, monthly_income, num_dependents, loan_tenure, interest_rate,
                outstanding_loan, collection_attempts,
                emi_to_income, collateral_coverage, default_severity
            ]])

            # Scale for KMeans
            input_scaled = scaler.transform(input_array)
            cluster_id = model.predict(input_scaled)[0]
            segment_label = segment_names.get(cluster_id, "Unknown Segment")

            # Risk score: proxy from engineered features
            # Normalize a composite risk score from key signals
            dpd_score = min(days_past_due / 180, 1.0)
            emi_score = min(emi_to_income / 1.0, 1.0)
            collateral_score = max(1 - collateral_coverage, 0)
            missed_score = min(missed_payments / 12, 1.0)
            risk_score = round(0.35 * dpd_score + 0.30 * missed_score + 0.20 * emi_score + 0.15 * collateral_score, 4)

            strategy, risk_category = assign_recovery_strategy(risk_score, days_past_due)

            # Store in session
            st.session_state["borrower_details"] = {
                "first_name": first_name,
                "last_name": last_name,
                "gender": gender,
                "age": age,
                "monthly_income": monthly_income,
                "num_dependents": num_dependents,
                "loan_type": loan_type,
                "loan_amount": loan_amount,
                "loan_tenure": loan_tenure,
                "interest_rate": interest_rate,
                "outstanding_loan": outstanding_loan,
                "collection_attempts": collection_attempts,
                "missed_payments": missed_payments,
                "days_past_due": days_past_due,
                "collateral_value": collateral_value,
                "monthly_emi": monthly_emi,
                "emi_to_income": emi_to_income,
                "collateral_coverage": collateral_coverage,
                "default_severity": default_severity,
                "risk_score": risk_score,
                "risk_category": risk_category,
                "strategy": strategy,
                "segment_label": segment_label,
                "custom_scheme": custom_scheme,
            }

            # Risk color
            if risk_score > 0.75:
                risk_color = "#C0392B"
            elif risk_score >= 0.25:
                risk_color = "#D4913A"
            else:
                risk_color = "#27AE60"

            st.markdown("<hr style='border-color:#2E4A7A; margin:24px 0;'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")

            r1, r2 = st.columns(2)
            card_base = (
                "background:rgba(245,248,255,0.96); border-radius:14px; padding:22px; "
                "min-height:260px; border:1.5px solid #1B3A6B; box-shadow:0 2px 14px rgba(10,22,46,0.10);"
            )

            with r1:
                st.markdown(
                    f"""
                    <div style="{card_base}">
                        <h3 style="color:#0F1F3D; font-size:1.5em; margin-bottom:10px;">
                            {first_name} {last_name}
                        </h3>
                        <p style="color:#1B3A6B;"><b>Age:</b> {age} &nbsp;|&nbsp; <b>Gender:</b> {gender}</p>
                        <p style="color:#1B3A6B;"><b>Loan Type:</b> {loan_type}</p>
                        <p style="color:#1B3A6B;"><b>Loan Amount:</b> ₹{loan_amount:,.0f}</p>
                        <p style="color:#1B3A6B;"><b>Monthly EMI:</b> ₹{monthly_emi:,.0f}</p>
                        <p style="color:#1B3A6B;"><b>Outstanding Loan:</b> ₹{outstanding_loan:,.0f}</p>
                        <p style="color:#1B3A6B;"><b>Borrower Segment:</b> {segment_label}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(
                    f"""
                    <div style="{card_base} border:2px solid {risk_color};">
                        <p style="color:#0F1F3D; font-weight:600; margin-bottom:4px;">Predicted Risk Score</p>
                        <p style="font-size:2.2em; font-weight:800; color:{risk_color}; margin:0 0 10px 0;">
                            {risk_score:.0%}
                        </p>
                        <p style="color:#0F1F3D;"><b>Risk Category:</b>
                            <span style="color:{risk_color}; font-weight:700;"> {risk_category}</span>
                        </p>
                        <p style="color:#0F1F3D; font-weight:600; margin-top:10px;">Recommended Strategy</p>
                        <p style="color:{risk_color}; font-size:0.97em;">{strategy}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<p style='color:#A8C0E0; margin-top:14px;'>"
                "View detailed analytics in the <b>Recovery Insights</b> page.</p>",
                unsafe_allow_html=True,
            )

            # PDF generation
            def generate_pdf(details):
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36,
                                        topMargin=36, bottomMargin=36)
                styles = getSampleStyleSheet()
                elements = []

                borrower_id = generate_borrower_id(
                    str(details.get("loan_type", "GEN")),
                    str(details.get("first_name", "X")),
                    str(details.get("last_name", "X")),
                )
                risk_cat_clean = remove_emoji(details["risk_category"])
                strategy_clean = remove_emoji(details["strategy"])
                rs = details["risk_score"]
                if rs > 0.75:
                    pdf_risk_color = "#C0392B"
                elif rs >= 0.25:
                    pdf_risk_color = "#D4913A"
                else:
                    pdf_risk_color = "#27AE60"

                elements.append(Paragraph("<b>Smart Loan Recovery System — Borrower Risk Report</b>", styles["Title"]))
                elements.append(Spacer(1, 14))
                elements.append(Paragraph(f"<b>Borrower ID:</b> {borrower_id}", styles["Normal"]))
                elements.append(Spacer(1, 10))

                table_data = [
                    ["Full Name", f"{details['first_name']} {details['last_name']}", "Age", details["age"]],
                    ["Gender", details["gender"], "Loan Type", details.get("loan_type", "—")],
                    ["Monthly Income (INR)", f"INR {details['monthly_income']:,}", "Loan Amount (INR)", f"INR {details['loan_amount']:,}"],
                    ["Outstanding Loan (INR)", f"INR {details['outstanding_loan']:,}", "Loan Tenure (months)", details["loan_tenure"]],
                    ["Interest Rate (%)", details["interest_rate"], "Collateral Value (INR)", f"INR {details['collateral_value']:,}"],
                    ["Missed Payments", details["missed_payments"], "Days Past Due", details["days_past_due"]],
                    ["Monthly EMI (INR)", f"INR {details['monthly_emi']:,.0f}", "EMI to Income Ratio", f"{details['emi_to_income']*100:.2f}%"],
                    ["Collateral Coverage", f"{details['collateral_coverage']*100:.2f}%", "Borrower Segment", details.get("segment_label", "—")],
                ]
                t = Table(table_data, colWidths=[130, 120, 130, 120])
                t.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                    ("BACKGROUND", (2, 0), (2, -1), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ]))
                elements.append(t)
                elements.append(Spacer(1, 16))
                elements.append(Paragraph(
                    f"<b>Predicted Risk Score:</b> <font color='{pdf_risk_color}'>{rs*100:.1f}%</font>", styles["Normal"]
                ))
                elements.append(Paragraph(
                    f"<b>Risk Category:</b> <font color='{pdf_risk_color}'>{risk_cat_clean}</font>", styles["Normal"]
                ))
                elements.append(Spacer(1, 8))
                elements.append(Paragraph("<b>Recommended Recovery Strategy:</b>", styles["Normal"]))
                elements.append(Paragraph(f"<font color='{pdf_risk_color}'>{strategy_clean}</font>", styles["BodyText"]))
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(
                    f"<font size=8 color=grey>Generated by Smart Loan Recovery System | ID: {borrower_id}</font>",
                    styles["Normal"]
                ))

                doc.build(elements)
                pdf_bytes = buffer.getvalue()
                buffer.close()
                return pdf_bytes, borrower_id

            st.info(f"Download the PDF report for {first_name} {last_name}.")
            pdf_bytes, borrower_id = generate_pdf(st.session_state["borrower_details"])
            st.download_button(
                label="Download Borrower PDF Report",
                data=pdf_bytes,
                file_name=f"borrower_report_{borrower_id}.pdf",
                mime="application/pdf",
            )

# ================================================
# PAGE 3: RECOVERY INSIGHTS
# ================================================
elif page == "Recovery Insights":
    if "borrower_details" not in st.session_state:
        risk_color_border = "#2E4A7A"
    else:
        rs = st.session_state["borrower_details"].get("risk_score", 0)
        if rs > 0.75:
            risk_color_border = "#C0392B"
        elif rs >= 0.25:
            risk_color_border = "#D4913A"
        else:
            risk_color_border = "#27AE60"

    # Background
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url('https://images.unsplash.com/photo-1597304055607-41e701431888?q=80&w=870&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
            background-size: cover;
            background-position: center;
        }}
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(8, 20, 45, 0.72);
            z-index: 0;
            pointer-events: none;
        }}
        [data-testid="stVerticalBlock"] {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background:rgba(10,22,46,0.90); border-radius:16px; padding:22px 28px;
                    margin-bottom:22px; border:2px solid {risk_color_border};">
            <h2 style="color:#4A90D9; margin-bottom:4px; font-size:2em;">Recovery Insights Dashboard</h2>
            <p style="color:#A8C0E0; margin:0; font-size:1.05em;">
                Borrower risk analytics, charts, and segment information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "borrower_details" not in st.session_state:
        st.warning("No prediction found. Please go to Smart Risk Predictor and submit borrower details first.")
        st.stop()

    details = st.session_state["borrower_details"]
    rs = details["risk_score"]

    if rs > 0.75:
        key_color = "#C0392B"
    elif rs >= 0.25:
        key_color = "#D4913A"
    else:
        key_color = "#27AE60"

    # Borrower summary table
    loan_type_disp = details.get("loan_type", "—")
    scheme_disp = "Yes" if details.get("custom_scheme", False) else "No"

    st.markdown(
        f"""
        <table style="width:100%; background:rgba(10,22,46,0.90); color:#E8F0FE;
                      border-radius:14px; border:1.5px solid {key_color};
                      border-collapse:collapse; margin-bottom:20px; font-size:1.02em;">
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9; background:rgba(74,144,217,0.08); width:30%;">Full Name</th>
                <td style="padding:9px 14px;">{details['first_name']} {details['last_name']}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Age</th>
                <td style="padding:9px 14px;">{details['age']}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Gender</th>
                <td style="padding:9px 14px;">{details['gender']}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Loan Type</th>
                <td style="padding:9px 14px;">{loan_type_disp}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Borrower Segment</th>
                <td style="padding:9px 14px;">{details.get('segment_label', '—')}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Loan Amount</th>
                <td style="padding:9px 14px;">₹{details['loan_amount']:,}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Monthly EMI</th>
                <td style="padding:9px 14px;">₹{details['monthly_emi']:,}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Outstanding Loan</th>
                <td style="padding:9px 14px;">₹{details['outstanding_loan']:,}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Risk Category</th>
                <td style="padding:9px 14px; color:{key_color}; font-weight:700;">{details['risk_category']}</td></tr>
            <tr><th style="padding:9px 14px; text-align:left; color:#4A90D9;">Strategy</th>
                <td style="padding:9px 14px; color:{key_color};">{details['strategy']}</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h3 style='color:#4A90D9; margin-bottom:10px;'>Analytical Visuals &amp; Key Metrics</h3>",
        unsafe_allow_html=True,
    )

    # --- Chart 1: Engineered Feature Percentages ---
    feat_bar = pd.DataFrame({
        "Feature": ["EMI to Income Ratio (%)", "Collateral Coverage (%)"],
        "Value": [details["emi_to_income"] * 100, details["collateral_coverage"] * 100],
    })
    fig_bar = px.bar(
        feat_bar, x="Feature", y="Value", color="Feature",
        title="Engineered Feature Percentages",
        text="Value", labels={"Value": "Percentage (%)"},
        color_discrete_sequence=["#4A90D9", "#27AE60"],
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="inside", marker_line_width=0)
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F0FE", title_font_color="#4A90D9",
        showlegend=False, height=380, margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        f"EMI to Income Ratio: {details['emi_to_income']*100:.2f}%. "
        f"Collateral Coverage: {details['collateral_coverage']*100:.2f}%. "
        "Higher EMI/Income ratio or lower Collateral Coverage indicates greater borrower risk."
    )

    # --- Chart 2: Payment History Donut ---
    if details["loan_tenure"] > 0:
        paid = max(details["loan_tenure"] - details["missed_payments"], 0)
        donut_df = pd.DataFrame({
            "Status": ["Paid", "Missed"],
            "Count": [paid, details["missed_payments"]],
        })
        fig_donut = px.pie(
            donut_df, names="Status", values="Count",
            title="Payment History (Paid vs Missed)", hole=0.55,
            color="Status", color_discrete_map={"Paid": "#27AE60", "Missed": "#C0392B"},
        )
        fig_donut.update_traces(textinfo="percent+label+value", textfont_size=15)
        fig_donut.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F0FE", title_font_color="#4A90D9", height=400,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption(f"Missed Payments: {details['missed_payments']} out of {details['loan_tenure']} total EMIs.")

    # --- Chart 3: Loan vs Collateral Pie ---
    total_exposure = details["loan_amount"] + details["collateral_value"]
    if total_exposure > 0:
        pie_df = pd.DataFrame({
            "Type": ["Loan Amount", "Collateral Value"],
            "Amount": [
                details["loan_amount"] / total_exposure * 100,
                details["collateral_value"] / total_exposure * 100,
            ],
        })
        fig_pie = px.pie(
            pie_df, names="Type", values="Amount",
            title="Loan vs Collateral Value (% of Total Exposure)",
            hole=0.4, color="Type",
            color_discrete_map={"Loan Amount": "#1B3A6B", "Collateral Value": "#4A90D9"},
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=15)
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F0FE", title_font_color="#4A90D9", height=400,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(
            f"Loan: {details['loan_amount']/total_exposure*100:.1f}%, "
            f"Collateral: {details['collateral_value']/total_exposure*100:.1f}% of total exposure."
        )

    # Collateral coverage alert
    if details["collateral_coverage"] < 1:
        st.warning("Collateral value is lower than the loan amount. This increases lender risk in case of default.")
    elif details["collateral_coverage"] == 1:
        st.info("Collateral matches the loan amount — basic security with no margin for error.")
    else:
        st.success("Collateral exceeds the loan amount — strong security for the lender.")

    # --- Chart 4: Risk Score Gauge ---
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=details["risk_score"] * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Predicted Risk Score (%)", "font": {"size": 20, "color": "#4A90D9"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4A90D9"},
            "bar": {"color": key_color, "thickness": 0.28},
            "bgcolor": "rgba(10,22,46,0.80)",
            "borderwidth": 2,
            "bordercolor": "#2E4A7A",
            "steps": [
                {"range": [0, 25], "color": "#1A4731"},
                {"range": [25, 75], "color": "#4A3210"},
                {"range": [75, 100], "color": "#4A1010"},
            ],
        },
        number={"suffix": "%", "font": {"color": key_color, "size": 30}},
    ))
    fig_gauge.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F0FE",
        height=420,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption(
        f"Predicted default risk: {details['risk_score']*100:.1f}%. "
        "Higher values indicate greater likelihood of default."
    )

    # --- Insights Summary ---
    st.markdown(
        """
        <div style="background:rgba(245,248,255,0.96); border-radius:14px; padding:18px 22px;
                    margin-top:14px; border:1.5px solid #1B3A6B;">
            <h3 style="color:#0F1F3D; margin-bottom:10px;">Insights Summary</h3>
        """,
        unsafe_allow_html=True,
    )

    insights = []
    if details["emi_to_income"] is not None:
        if details["emi_to_income"] > 0.5:
            insights.append("EMI burden exceeds 50% of monthly income — high repayment stress.")
        elif details["emi_to_income"] > 0.35:
            insights.append("EMI burden is moderate (35–50% of income) — monitor closely.")
        else:
            insights.append("EMI burden is low relative to income — manageable repayment.")
    if details["missed_payments"] is not None:
        if details["missed_payments"] >= 4:
            insights.append(f"Missed {details['missed_payments']} EMIs — indicates growing payment fatigue.")
        elif details["missed_payments"] > 0:
            insights.append(f"Missed {details['missed_payments']} EMI(s) recently — early warning sign.")
        else:
            insights.append("No missed EMIs — good repayment track record.")
    if details["collateral_value"] == 0:
        insights.append("No collateral provided — fully unsecured risk exposure.")
    elif details["collateral_coverage"] < 1:
        insights.append("Collateral is less than loan amount — higher risk in case of default.")
    elif details["collateral_coverage"] >= 1:
        insights.append("Collateral meets or exceeds loan amount — good security for lender.")
    if details["days_past_due"] >= 90:
        insights.append("DPD ≥ 90 days — borrower is in the critical zone. Consider legal action.")
    elif details["days_past_due"] > 0:
        insights.append(f"Days Past Due: {details['days_past_due']} — follow up required.")

    for bullet in insights:
        st.markdown(f"<p style='color:#1B3A6B; margin:6px 0;'>▸ {bullet}</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================================================
# PAGE 4: DASHBOARD
# ================================================
elif page == "Dashboard":
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0F1F3D;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background:rgba(10,22,46,0.90); border-radius:16px; padding:22px 28px;
                    margin-bottom:22px; border:2px solid #2E4A7A;">
            <h2 style="color:#FFFFFF; margin-bottom:4px; font-size:2em;">📊 Loan Recovery Dashboard</h2>
            <p style="color:#FFFFFF; margin:0; font-size:1.05em;">
                Comprehensive analytics and visual overview of the Smart Loan Recovery System.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with open("dashboard/smart_loan_recovery_dashboard.html", "r", encoding="utf-8") as f:
            dashboard_html = f.read()

        custom_dashboard_style = """
        <style>
            :root {
                --color-text-primary: #E8F0FE;
                --color-text-secondary: #A8C0E0;
                --color-background-primary: rgba(15, 31, 61, 0.95);
                --color-background-secondary: rgba(255, 255, 255, 0.06);
                --color-border-tertiary: rgba(74, 144, 217, 0.25);
                --border-radius-lg: 16px;
                --border-radius-md: 12px;
                --font-sans: 'DM Sans', sans-serif;
            }

            body {
                background-color: #0F1F3D;
                color: #E8F0FE;
                margin: 0;
                padding: 16px;
            }

            .sr-only {
                display: none !important;
            }

            .slrs,
            .slrs h1,
            .slrs h2,
            .slrs h3,
            .slrs p,
            .slrs div,
            .slrs span {
                color: #E8F0FE;
            }

            .metric-label,
            .chart-title,
            .stat-name,
            .dl-label,
            .rb-label,
            .metric-sub,
            .stat-pct,
            .summary-card p {
                color: #A8C0E0 !important;
            }

            canvas {
                background: transparent !important;
            }
        </style>

        <script>
        window.addEventListener("load", function () {
            setTimeout(function () {
                if (window.Chart) {
                    Chart.defaults.color = "#FF4D8D";
                    Chart.defaults.borderColor = "rgba(255,255,255,0.08)";

                    const chartIds = ["c1", "c2", "c3", "c4"];

                    chartIds.forEach(function(id) {
                        const canvas = document.getElementById(id);
                        if (!canvas) return;

                        const chart = Chart.getChart(canvas);
                        if (!chart) return;

                        if (chart.options.scales) {
                            Object.keys(chart.options.scales).forEach(function(axisKey) {
                                const axis = chart.options.scales[axisKey];
                                if (!axis.ticks) axis.ticks = {};
                                axis.ticks.color = "#FF4D8D";
                            });
                        }

                        if (chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {
                            chart.options.plugins.legend.labels.color = "#FF4D8D";
                        }

                        chart.update();
                    });
                }
            }, 800);
        });
        </script>
        """

        dashboard_html = custom_dashboard_style + dashboard_html
        components.html(dashboard_html, height=1800, scrolling=True)

    except FileNotFoundError:
        st.error("Dashboard file not found. Please check the file path.")
