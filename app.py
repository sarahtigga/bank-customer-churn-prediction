import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")

st.title("Customer Churn Prediction")
st.subheader("Predictive Modeling and Risk Scoring")
st.write(
    "This dashboard uses a Random Forest model to predict customer churn, "
    "evaluate model performance, and categorize customers by churn risk."
)


@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv.csv")


@st.cache_resource
def train_model(data):
    df = data.copy()

    df = df.drop(
        columns=[
            col for col in ["CustomerId", "Surname"]
            if col in df.columns
        ],
        errors="ignore"
    )

    if "Exited" not in df.columns:
        raise ValueError("The dataset must contain an 'Exited' column.")

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Exited", axis=1)
    y = df_encoded["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, X.columns, y_test, y_pred, y_prob


try:
    df = load_data()
    model, feature_names, y_test, y_pred, y_prob = train_model(df)
except FileNotFoundError:
    st.error(
        "European_Bank.csv.csv was not found. "
        "Place the CSV file in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load or train the model: {e}")
    st.stop()


roc_score = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

# DATASET OVERVIEW
st.header("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{len(df):,}")
c2.metric("Total Features", f"{df.shape[1] - 1}")
c3.metric("Test Customers", f"{len(y_test):,}")
c4.metric("ROC-AUC Score", f"{roc_score:.3f}")

# MODEL PERFORMANCE
st.header("2. Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", f"{accuracy:.1%}")
c2.metric("Churn Precision", f"{report['1']['precision']:.1%}")
c3.metric("Churn Recall", f"{report['1']['recall']:.1%}")

st.caption(
    "Precision measures how often predicted churners actually churned. "
    "Recall measures how many actual churners the model identified."
)

# CONFUSION MATRIX AND ROC CURVE
st.header("3. Model Evaluation")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Stay", "Predicted Churn"],
        yticklabels=["Actual Stay", "Actual Churn"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        fpr,
        tpr,
        label=f"Random Forest (AUC = {roc_score:.3f})"
    )
    ax.plot([0, 1], [0, 1], "--", label="Baseline")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# FEATURE IMPORTANCE
st.header("4. Top 10 Feature Importance")

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=feat_df,
    x="Importance",
    y="Feature",
    ax=ax
)

ax.set_title("Most Important Features in Churn Prediction")
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# CHURN PROBABILITY DISTRIBUTION
st.header("5. Churn Probability Distribution")

prob_df = pd.DataFrame({
    "Churn Probability": y_prob
})

fig, ax = plt.subplots(figsize=(8, 4))

sns.histplot(
    data=prob_df,
    x="Churn Probability",
    bins=20,
    kde=True,
    color="purple",
    ax=ax
)

ax.set_xlabel("Predicted Churn Probability")
ax.set_ylabel("Number of Customers")
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# CUSTOMER RISK SCORING
st.header("6. Customer Risk Scoring")

risk_df = pd.DataFrame({
    "Actual Outcome": np.where(
        np.asarray(y_test) == 1,
        "Churned",
        "Stayed"
    ),
    "Churn Probability": y_prob
})

risk_df["Risk Segment"] = pd.cut(
    risk_df["Churn Probability"],
    bins=[-0.001, 0.30, 0.60, 1.001],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

risk_counts = risk_df["Risk Segment"].value_counts()

c1, c2, c3 = st.columns(3)

c1.metric("Low-Risk Customers", int(risk_counts.get("Low Risk", 0)))
c2.metric("Medium-Risk Customers", int(risk_counts.get("Medium Risk", 0)))
c3.metric("High-Risk Customers", int(risk_counts.get("High Risk", 0)))

st.caption(
    "Risk categories use illustrative thresholds: Low Risk up to 30%, "
    "Medium Risk above 30% through 60%, and High Risk above 60%."
)

st.subheader("Customer Risk Scores")

show_all = st.checkbox("Show all test customers", value=False)

display_df = risk_df.copy()
display_df["Churn Probability"] = display_df["Churn Probability"].round(4)

if not show_all:
    display_df = display_df.sort_values(
        "Churn Probability",
        ascending=False
    ).head(20)

st.dataframe(display_df, use_container_width=True)

st.download_button(
    label="Download Customer Risk Scores",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_churn_risk_scores.csv",
    mime="text/csv"
)

# CLASSIFICATION REPORT
st.header("7. Classification Report")

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)

st.subheader("Classification Report Heatmap")

heatmap_df = report_df.loc[
    ["0", "1"],
    ["precision", "recall", "f1-score"]
]

fig, ax = plt.subplots(figsize=(7, 3))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    ax=ax
)

ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Customer Class")
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# KEY INSIGHTS
st.header("8. Key Insights")

st.markdown(f"""
- The Random Forest model achieved a **ROC-AUC score of {roc_score:.3f}**.
- Overall prediction accuracy is **{accuracy:.1%}** on the test dataset.
- The model identified **{report['1']['recall']:.1%}** of actual churners at its default threshold.
- Feature importance highlights the variables most influential to the fitted model.
- Risk scores can help prioritize customer-retention efforts.
""")

st.info(
    "Model configuration: Random Forest with 150 trees, an 80/20 "
    "stratified train-test split, random_state=42, and balanced class weights."
)
