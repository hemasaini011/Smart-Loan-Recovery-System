# Smart Loan Recovery System

A professional, AI-powered Streamlit web application for loan risk assessment, borrower segmentation, and recovery strategy recommendations. Designed for financial institutions and analysts who need fast, reliable, and explainable loan default predictions — with downloadable reports and an interactive analytics dashboard.

---

## Project Description

The **Smart Loan Recovery System (SLRS)** is a machine learning-based web application that helps financial teams predict the likelihood of loan default for individual borrowers and recommends a tailored recovery strategy. By entering basic borrower and loan details, the system automatically calculates key financial indicators, assigns a risk score, and provides an actionable recovery plan — all within a clean, modern interface.

The system uses **K-Means clustering** for borrower segmentation and a **composite risk scoring model** built from engineered features like EMI-to-income ratio, days past due, collateral coverage, and missed payments. Results can be exported as a professional PDF report for documentation and follow-up.

---

## Features

- **Smart Risk Predictor**
  - Enter borrower details to instantly get a risk score and default probability
  - Loan type selector with auto-filled interest rates and tenures (Personal, Auto, Business, Home)
  - Auto-calculated fields: EMI, Days Past Due (DPD), and Collection Attempts
  - Custom loan scheme support for special interest rate and tenure overrides
  - Risk categories: Low Risk, Medium Risk, High Risk, Critical Risk
  - Tailored recovery strategy recommendation based on risk score and DPD
  - Downloadable, color-coded PDF report with a unique Borrower ID

- **Recovery Insights Dashboard**
  - Visual analytics for the predicted borrower: EMI vs Income, Collateral Coverage, Payment History
  - Risk Score Gauge chart using Plotly
  - Plain-language Insights Summary based on business logic
  - Borrower Segment label from K-Means clustering

- **Overview Page**
  - App introduction, feature highlights, and usage guide
  - Clean card-based layout with icons

- **Analytics Dashboard**
  - Embedded HTML dashboard with Chart.js visuals for portfolio-level loan recovery analytics
  - Comprehensive overview of borrower segments and recovery trends

- **Modern UI/UX**
  - Dark navy theme with custom CSS for sidebar, buttons, inputs, and cards
  - Responsive layout using Streamlit columns and styled containers
  - Flaticon icons and Unsplash backgrounds for a polished look

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| Web Framework | Streamlit, streamlit-option-menu |
| Machine Learning | scikit-learn (K-Means, StandardScaler) |
| Data Processing | NumPy, Pandas |
| Visualization | Plotly, Matplotlib, Chart.js (HTML dashboard) |
| PDF Generation | ReportLab |
| Model Persistence | Pickle |
| Frontend Styling | Custom CSS (via Streamlit Markdown) |
| Language | Python 3.x |

---

## File Structure

```
Smart-Loan-Recovery-System/
│
├── app1.py                          # Main Streamlit application
│
├── models/
│   ├── kmeans.pkl                   # Trained K-Means clustering model
│   ├── scaler.pkl                   # StandardScaler for feature normalization
│   ├── features.pkl                 # Feature list used during training
│   ├── segment_names.pkl            # Human-readable cluster/segment labels
│   └── reference_data.pkl           # Reference data for analytics
│
├── dataset/
│   ├── loan-recovery.csv            # Primary loan recovery dataset
│   └── reference_data.csv           # Reference data (CSV format)
│
├── dashboard/
│   └── smart_loan_recovery_dashboard.html   # Embedded analytics dashboard (Chart.js)
│
├── notebook/
│   └── smart_loan_recovery_basic.ipynb      # Model training and EDA notebook
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/Smart-Loan-Recovery-System.git
cd Smart-Loan-Recovery-System
```

**2. (Optional) Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify Model Files**

Make sure the following files are present inside the `models/` folder:
- `kmeans.pkl`
- `scaler.pkl`
- `features.pkl`
- `segment_names.pkl`

If they are missing, run the Jupyter notebook `notebook/smart_loan_recovery_basic.ipynb` to regenerate them.

---

## How to Run the Project

```bash
streamlit run app1.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How to Use the Application

1. Open the app and land on the **Overview** page for a quick introduction.
2. Navigate to **Smart Risk Predictor** from the sidebar.
3. Select a **Loan Type** — the system will auto-fill default interest rate and tenure.
4. Fill in borrower details: name, age, income, loan amount, collateral, missed payments, etc.
5. Fields like **EMI**, **Days Past Due**, and **Collection Attempts** are calculated automatically.
6. Click **Predict Risk & Strategy** to get the risk score and recommended recovery action.
7. Go to **Recovery Insights** to view detailed charts and an insights summary for the borrower.
8. Download the **PDF Report** directly from the prediction results.
9. Visit the **Dashboard** tab for a portfolio-level visual analytics overview.

---

## Model and Logic

### Borrower Segmentation
- **Algorithm:** K-Means Clustering
- **Features Used:** Age, Monthly Income, Number of Dependents, Loan Tenure, Interest Rate, Outstanding Loan, Collection Attempts, EMI-to-Income Ratio, Collateral Coverage, Default Severity
- **Segments:** 4 distinct borrower clusters with human-readable labels (e.g., Low-Risk Regular Payer, High-Risk Defaulter, etc.)

### Risk Score Calculation
A composite risk score is computed from weighted engineered features:

| Signal | Weight |
|---|---|
| Days Past Due (DPD) | 35% |
| Missed Payments | 30% |
| EMI-to-Income Ratio | 20% |
| Collateral Coverage Gap | 15% |

### Recovery Strategy Assignment

| Risk Score | DPD | Category | Strategy |
|---|---|---|---|
| > 90% | ≥ 90 days | Critical Risk | Legal proceedings, collateral seizure, external agencies |
| > 90% | < 90 days | High Risk | Pre-litigation warning, restructuring, senior escalation |
| 75% – 90% | Any | High Risk | One-time settlement, revised repayment, pre-litigation notice |
| 25% – 75% | Any | Medium Risk | Soft recovery calls, EMI restructuring, behavior analysis |
| < 25% | Any | Low Risk | Automated reminders, financial advisory nudges |

---

## Output and Functionality

- **Prediction Result Card:** Displays borrower profile, risk score (color-coded), risk category, and recovery strategy.
- **PDF Report:** A professionally formatted A4 document with borrower details, risk summary, and strategy — downloadable with a unique Borrower ID.
- **Recovery Insights:** Four interactive Plotly charts (EMI Burden, Payment History, Collateral Pie, Risk Gauge) plus a plain-language insights panel.
- **Analytics Dashboard:** An embedded HTML dashboard showing portfolio-level metrics using Chart.js.

---

## Author

**Hema Saini**

For questions, support, or demo requests, feel free to reach out via the Contact Us page in the application.

---

## License

This project is developed as an academic/final year project. All rights reserved by the author.
