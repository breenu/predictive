# Customer Churn Analysis (Case Study 2)

IBM Telco Customer Churn: descriptive analytics, Random Forest prediction, prescriptive recommendations, and a Streamlit dashboard for review.

## Setup

```bash
pip install -r requirements.txt
```

Optional: place `telco_customer_churn.csv` in the `data` folder. If it is missing, the app downloads the dataset from a public GitHub mirror on first run.

## Run the dashboard

```bash
streamlit run app.py
```

If `streamlit` is not on your PATH (common on Windows), use:

```bash
python -m streamlit run app.py
```

Your browser opens a local URL (typically `http://localhost:8501`). Use the tabs for overview, EDA, model metrics, recommendations, and a simple churn probability demo.

## Project layout

| File | Purpose |
|------|---------|
| `churn_analysis.py` | Load data, clean, EDA plots, train Random Forest, evaluate, prescriptive text |
| `app.py` | Streamlit UI |
| `requirements.txt` | Python dependencies |
