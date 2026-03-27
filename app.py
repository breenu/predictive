import matplotlib.pyplot as plt
import streamlit as st

from churn_analysis import figures_for_dashboard, predict_proba_from_inputs, train_pipeline_core

st.set_page_config(page_title="Telco Churn Analysis", layout="wide")

ANALYTICS_CLASSIFICATION = """
### How this problem fits business analytics

- **Descriptive analytics** summarizes what happened: overall churn rate, churn by contract type, tenure bands, and bill amounts—so we see historical patterns in the data.
- **Diagnostic analytics** explains *why* those patterns appear: for example, month-to-month contracts and higher monthly charges correlate with churn, and the model’s feature importances point to drivers.
- **Predictive analytics** forecasts *who* is likely to churn next: the Random Forest assigns a churn probability to each customer using past behavior.
- **Prescriptive analytics** turns those insights into actions: retention offers, contract upgrades, pricing reviews, and prioritized outreach lists.

Together, these layers answer “what happened,” “why,” “who will churn,” and “what should we do.”
"""


@st.cache_resource
def load_pipeline():
    return train_pipeline_core()


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("Case Study 2: Customer Churn Analysis")
    st.caption("Telecom Customer Churn — descriptive, predictive, and prescriptive analytics")

    with st.spinner("Loading data and training model (first run may take a minute)…"):
        result = load_pipeline()
        fig_pack = figures_for_dashboard(result["df_raw"], result["metrics"], result["importance_df"])

    metrics = result["metrics"]
    bundle = result["bundle"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Descriptive analytics", "Predictive model", "Prescriptive actions", "Try a prediction"]
    )

    with tab1:
        st.markdown(ANALYTICS_CLASSIFICATION)
        st.subheader("Dataset snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", f"{result['n_samples']:,}")
        c2.metric("Churn rate", f"{result['churn_rate']:.1%}")
        c3.metric("ROC-AUC (test)", f"{metrics['roc_auc']:.4f}")
        st.info(
            "Class balance: churn is typically **lower** than the retained class; "
            "the model uses balanced class weights so the minority class is not ignored."
        )

    with tab2:
        st.subheader("Churn trends (descriptive statistics)")
        st.write(
            "Explore how churn varies by contract, tenure, and monthly charges, and how numeric features correlate."
        )
        eda = fig_pack["eda_figures"]
        for name, fig in eda.items():
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            show_fig(fig)

    with tab3:
        st.subheader("Random Forest classifier")
        st.write(
            "**Hyperparameters:** 200 trees, max depth 8, min samples per leaf 4, "
            "`class_weight='balanced'`, random state 42. "
            "Numeric features are standardized; categoricals are one-hot encoded."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("Precision", f"{metrics['precision']:.4f}")
        m3.metric("Recall", f"{metrics['recall']:.4f}")
        m4.metric("F1", f"{metrics['f1']:.4f}")
        st.markdown("**Classification report (test set)**")
        st.text(metrics["classification_report"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Confusion matrix**")
            show_fig(fig_pack["confusion_fig"])
        with col_b:
            st.markdown("**ROC curve**")
            show_fig(fig_pack["roc_fig"])
        st.markdown("**Top feature importances**")
        show_fig(fig_pack["importance_fig"])

    with tab4:
        st.subheader("Recommended actions")
        st.write("Based on descriptive patterns and model drivers:")
        for i, rec in enumerate(result["prescriptive"], 1):
            st.markdown(f"{i}. {rec}")

    with tab5:
        st.subheader("Estimate churn probability")
        st.write(
            "Adjust a few **key fields**; other features use typical (median) values from the training data. "
            "This is a simplified demo, not a production API."
        )
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
        total = st.slider("Total charges ($)", 0.0, 9000.0, 500.0)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            index=0,
        )
        if st.button("Predict churn probability"):
            p = predict_proba_from_inputs(
                bundle,
                tenure=float(tenure),
                monthly_charges=float(monthly),
                total_charges=float(total),
                contract=contract,
            )
            st.metric("Estimated P(churn)", f"{p:.1%}")
            if p >= 0.5:
                st.warning("Above 50% threshold — flag for retention outreach.")
            else:
                st.success("Below 50% threshold — lower priority for retention calls.")


if __name__ == "__main__":
    main()
