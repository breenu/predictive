"""
IBM Telco Customer Churn: load, EDA, modeling, and prescriptive recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TELCO_URLS = (
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/"
    "master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/"
    "master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
)


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load Telco churn CSV from local path or GitHub mirrors."""
    path = Path(csv_path) if csv_path else Path(__file__).resolve().parent / "data" / "telco_customer_churn.csv"
    if path.exists():
        return pd.read_csv(path)
    last_err: Exception | None = None
    for url in TELCO_URLS:
        try:
            return pd.read_csv(url)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("Could not load Telco dataset")


def _clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col = None
    if "TotalCharges" in out.columns:
        col = "TotalCharges"
    elif "Total Charges" in out.columns:
        col = "Total Charges"
    if col:
        out[col] = out[col].replace(r"^\s*$", np.nan, regex=True)
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix (raw, before encoding) and binary target."""
    df = _clean_total_charges(df)
    target_col = "Churn Value" if "Churn Value" in df.columns else "Churn"
    y = df[target_col]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().map({"Yes": 1, "No": 0, "True": 1, "False": 0})
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    cols_to_drop = [
        "CustomerID",
        "customerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
        target_col,
    ]
    actual = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=actual)
    return X, y


@dataclass
class ModelBundle:
    model: RandomForestClassifier
    scaler: StandardScaler
    feature_columns: pd.Index
    numeric_cols: pd.Index
    X_train_encoded: pd.DataFrame
    median_encoded_row: pd.Series
    raw_numeric_medians: pd.Series


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ModelBundle, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode, split, scale, train RF. Returns bundle, metrics dict, y_test, y_pred, y_prob, numeric_cols."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    if len(numeric_cols):
        X_train_s[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_s[numeric_cols] = scaler.transform(X_test[numeric_cols])

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    y_prob = rf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    median_encoded_row = X_train_s.median()
    raw_numeric_medians = X_train[numeric_cols].median() if len(numeric_cols) else pd.Series(dtype=float)

    bundle = ModelBundle(
        model=rf,
        scaler=scaler,
        feature_columns=X_train_s.columns,
        numeric_cols=pd.Index(numeric_cols),
        X_train_encoded=X_train_s,
        median_encoded_row=median_encoded_row,
        raw_numeric_medians=raw_numeric_medians,
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": roc_curve(y_test, y_prob)[0],
        "tpr": roc_curve(y_test, y_prob)[1],
        "thresholds": roc_curve(y_test, y_prob)[2],
    }
    return bundle, metrics, y_test.values, y_pred, y_prob, X_test_s


def feature_importance_df(model: RandomForestClassifier, columns: pd.Index, top_n: int = 15) -> pd.DataFrame:
    imp = pd.DataFrame({"Feature": columns, "Importance": model.feature_importances_})
    return imp.sort_values("Importance", ascending=False).head(top_n)


def build_eda_figures(df_raw: pd.DataFrame) -> dict[str, plt.Figure]:
    """Descriptive analytics plots (needs raw columns: Churn, Contract, tenure, MonthlyCharges)."""
    df = _clean_total_charges(df_raw.copy())
    churn_col = "Churn Value" if "Churn Value" in df.columns else "Churn"
    if churn_col not in df.columns:
        raise ValueError("Dataset must contain Churn or Churn Value column.")

    churn_binary = df[churn_col].map({"Yes": 1, "No": 0}) if df[churn_col].dtype == object else df[churn_col]
    churn_binary = pd.to_numeric(churn_binary, errors="coerce").fillna(0).astype(int)
    df = df.assign(_churn=churn_binary)

    sns.set_theme(style="whitegrid")
    figures: dict[str, plt.Figure] = {}

    # 1. Churn rate by contract
    if "Contract" in df.columns:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ct = df.groupby("Contract")["_churn"].mean().sort_values(ascending=False)
        sns.barplot(x=ct.index, y=ct.values, hue=ct.index, palette="viridis", ax=ax1, legend=False)
        ax1.set_ylabel("Churn rate")
        ax1.set_xlabel("Contract type")
        ax1.set_title("Churn rate by contract type")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha="right")
        fig1.tight_layout()
        figures["churn_by_contract"] = fig1

    # 2. Tenure bins
    if "tenure" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        df["_tenure_bin"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72, 100], labels=["0-12", "13-24", "25-48", "49-72", "73+"])
        tb = df.groupby("_tenure_bin", observed=True)["_churn"].mean()
        sns.barplot(x=tb.index.astype(str), y=tb.values, hue=tb.index.astype(str), ax=ax2, palette="crest", legend=False)
        ax2.set_ylabel("Churn rate")
        ax2.set_xlabel("Tenure (months)")
        ax2.set_title("Churn rate by tenure band")
        fig2.tight_layout()
        figures["churn_by_tenure"] = fig2

    # 3. Monthly charges by churn
    if "MonthlyCharges" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        plot_df = df[["MonthlyCharges", "_churn"]].copy()
        plot_df["_churn_lab"] = plot_df["_churn"].map({0: "Retained", 1: "Churned"})
        sns.boxplot(data=plot_df, x="_churn_lab", y="MonthlyCharges", hue="_churn_lab", palette="Set2", ax=ax3, legend=False)
        ax3.set_title("Monthly charges by churn status")
        ax3.set_xlabel("")
        fig3.tight_layout()
        figures["monthly_charges_box"] = fig3

    # 4. Numeric correlation heatmap (subset)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ("_churn",) and df[c].nunique() > 1][:12]
    if len(num_cols) >= 2:
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax4, linewidths=0.5)
        ax4.set_title("Correlation heatmap (numeric features)")
        fig4.tight_layout()
        figures["correlation_heatmap"] = fig4

    return figures


def plot_confusion_matrix_fig(cm: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_theme(style="whitegrid")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, annot_kws={"size": 12})
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Retained", "Churned"])
    ax.set_yticklabels(["Retained", "Churned"])
    fig.tight_layout()
    return fig


def plot_roc_fig(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_importance_fig(importance_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="magma",
        ax=ax,
        legend=False,
    )
    ax.set_title("Top features: churn drivers (Random Forest)")
    ax.set_xlabel("Relative importance")
    fig.tight_layout()
    return fig


def prescriptive_recommendations(
    df_raw: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> list[str]:
    """Business actions from EDA + model importances."""
    df = _clean_total_charges(df_raw.copy())
    churn_col = "Churn Value" if "Churn Value" in df.columns else "Churn"
    churn_binary = df[churn_col].map({"Yes": 1, "No": 0}) if df[churn_col].dtype == object else df[churn_col]
    churn_binary = pd.to_numeric(churn_binary, errors="coerce").fillna(0).astype(int)
    df = df.assign(_churn=churn_binary)

    recs: list[str] = []
    top_feats = importance_df["Feature"].head(8).tolist()

    if "Contract" in df.columns:
        mtm = df[df["Contract"].str.contains("Month", case=False, na=False)]["_churn"].mean()
        two_y = df[df["Contract"].astype(str).str.contains("Two", case=False, na=False)]["_churn"].mean()
        if pd.notna(mtm) and pd.notna(two_y):
            recs.append(
                f"Prioritize retention for month-to-month subscribers (observed churn ~{mtm:.1%} vs "
                f"~{two_y:.1%} for two-year contracts). Offer annual or two-year discounts to convert plan type."
            )

    if "tenure" in df.columns:
        low_t = df[df["tenure"] <= 12]["_churn"].mean()
        recs.append(
            f"Early tenure (≤12 months) shows higher churn risk (~{low_t:.1%}). "
            "Run onboarding check-ins and first-year loyalty perks."
        )

    if any("MonthlyCharges" in f or "TotalCharges" in f for f in top_feats):
        recs.append(
            "High monthly charges and total charges rank among top model drivers. "
            "Review pricing bundles, fee transparency, and competitive matching for at-risk high-bill customers."
        )

    if any("InternetService" in f or "OnlineSecurity" in f or "TechSupport" in f for f in top_feats):
        recs.append(
            "Service add-ons (security, support, fiber) matter in the model. "
            "Proactively offer tech support and security bundles to reduce voluntary churn."
        )

    if any("PaymentMethod" in f for f in top_feats):
        recs.append(
            "Payment method appears in churn patterns. Promote automatic bank transfer or card payments "
            "to reduce failed payments and involuntary churn."
        )

    recs.append(
        "Use the model's probability scores to build a ranked call list for retention teams, "
        "allocating higher-touch outreach to the top decile of predicted churn risk."
    )

    recs.append(
        "Limitations: findings are correlational; causal impact of interventions should be tested with A/B tests. "
        "Data is a single-period snapshot and may not reflect seasonality or competitive moves."
    )

    return recs


def predict_proba_from_inputs(
    bundle: ModelBundle,
    *,
    tenure: float,
    monthly_charges: float,
    total_charges: float,
    contract: str,
) -> float:
    """
    Single-row churn probability: median profile with overrides for tenure, charges, and contract.
    contract: 'Month-to-month', 'One year', 'Two year'
    """
    row = bundle.median_encoded_row.copy()
    raw = bundle.raw_numeric_medians.reindex(bundle.numeric_cols).fillna(0).copy()
    for c in bundle.numeric_cols:
        if c == "tenure":
            raw[c] = tenure
        elif c == "MonthlyCharges":
            raw[c] = monthly_charges
        elif "TotalCharges" in str(c) or "Total Charges" in str(c):
            raw[c] = total_charges

    raw_df = pd.DataFrame([[float(raw[c]) for c in bundle.numeric_cols]], columns=list(bundle.numeric_cols))
    scaled = bundle.scaler.transform(raw_df)[0]
    for i, c in enumerate(bundle.numeric_cols):
        if c in row.index:
            row[c] = scaled[i]

    for col in row.index:
        if str(col).startswith("Contract_"):
            row[col] = 0.0
    if contract == "One year" and "Contract_One year" in row.index:
        row["Contract_One year"] = 1.0
    elif contract == "Two year" and "Contract_Two year" in row.index:
        row["Contract_Two year"] = 1.0

    X = pd.DataFrame([row[bundle.feature_columns].values], columns=bundle.feature_columns)
    return float(bundle.model.predict_proba(X)[0, 1])


def figures_for_dashboard(
    df_raw: pd.DataFrame,
    metrics: dict[str, Any],
    importance_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build matplotlib figures for Streamlit (call each run so plots stay valid after plt.close)."""
    return {
        "eda_figures": build_eda_figures(df_raw),
        "confusion_fig": plot_confusion_matrix_fig(metrics["confusion_matrix"]),
        "roc_fig": plot_roc_fig(metrics["fpr"], metrics["tpr"], metrics["roc_auc"]),
        "importance_fig": plot_importance_fig(importance_df),
    }


def run_full_pipeline(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Load data, train model, prescriptive text, and all figures (for scripts/tests)."""
    df_raw = load_data(csv_path)
    X, y = prepare_features_and_target(df_raw)

    bundle, metrics, _, _, _, _ = train_random_forest(X, y)
    imp_df = feature_importance_df(bundle.model, bundle.feature_columns)
    recs = prescriptive_recommendations(df_raw, imp_df)
    figs = figures_for_dashboard(df_raw, metrics, imp_df)

    churn_rate = float(y.mean())
    n = len(y)

    return {
        "df_raw": df_raw,
        "X": X,
        "y": y,
        "bundle": bundle,
        "metrics": metrics,
        "importance_df": imp_df,
        "eda_figures": figs["eda_figures"],
        "confusion_fig": figs["confusion_fig"],
        "roc_fig": figs["roc_fig"],
        "importance_fig": figs["importance_fig"],
        "prescriptive": recs,
        "churn_rate": churn_rate,
        "n_samples": n,
    }


def train_pipeline_core(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Load data, train model—cacheable without matplotlib figures."""
    df_raw = load_data(csv_path)
    X, y = prepare_features_and_target(df_raw)
    bundle, metrics, _, _, _, _ = train_random_forest(X, y)
    imp_df = feature_importance_df(bundle.model, bundle.feature_columns)
    recs = prescriptive_recommendations(df_raw, imp_df)
    return {
        "df_raw": df_raw,
        "bundle": bundle,
        "metrics": metrics,
        "importance_df": imp_df,
        "prescriptive": recs,
        "churn_rate": float(y.mean()),
        "n_samples": len(y),
    }
