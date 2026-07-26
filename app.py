import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
import os

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main{
    padding-top:1rem;
}

.metric-card{
    background:#f8f9fa;
    padding:15px;
    border-radius:10px;
    border:1px solid #e6e6e6;
}

div[data-testid="metric-container"]{
    border:1px solid #d3d3d3;
    padding:12px;
    border-radius:10px;
    background-color:#fafafa;
}

.stButton>button{
    width:100%;
    height:45px;
    font-size:18px;
    border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================

st.title("📱 Mobile Price Classification Dashboard")

st.markdown("""
Predict the **price range of a mobile phone**
using multiple Machine Learning models.
""")

# ======================================================
# LOAD MODELS
# ======================================================

@st.cache_resource
def load_models():

    models = {

        "Logistic Regression":
            joblib.load("models/logistic_regression.pkl"),

        "Decision Tree":
            joblib.load("models/decision_tree.pkl"),

        "KNN":
            joblib.load("models/knn.pkl"),

        "Naive Bayes":
            joblib.load("models/naive_bayes.pkl"),

        "Random Forest":
            joblib.load("models/random_forest.pkl")

    }

    scaler = joblib.load("models/scaler.pkl")

    return models, scaler


models, scaler = load_models()

# ======================================================
# SIDEBAR HEADER
# ======================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/mobile-phone.png",
    width=80
)

st.sidebar.title("Configuration")

selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

input_method = st.sidebar.radio(
    "Choose Input Method",
    (
        "Sample Dataset",
        "Upload CSV",
        "Manual Input"
    )
)

st.sidebar.markdown("---")


# ======================================================
# INPUT METHODS
# ======================================================

df = None

# -------------------------------
# OPTION 1 : SAMPLE DATASET
# -------------------------------

if input_method == "Sample Dataset":

    st.header("📄 Sample Dataset")

    sample_size = st.slider(
        "Number of sample records",
        5,
        100,
        10
    )

    df = pd.read_csv("data/mobile_price.csv")

    if "price_range" in df.columns:
        df = df.drop(columns=["price_range"])

    df = df.head(sample_size)

    st.success(f"{sample_size} sample records loaded successfully.")


# -------------------------------
# OPTION 2 : UPLOAD CSV
# -------------------------------

elif input_method == "Upload CSV":

    st.header("📤 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # Remove target column if present
        if "price_range" in df.columns:
            df = df.drop(columns=["price_range"])

        # Remove id column if present
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        st.success("Dataset uploaded successfully.")

    else:

        st.info("Please upload a CSV file.")

        st.stop()


# -------------------------------
# OPTION 3 : MANUAL INPUT
# -------------------------------

else:

    st.header("✍ Manual Input")

    col1, col2 = st.columns(2)

    with col1:

        battery_power = st.number_input(
            "Battery Power",
            500,
            2000,
            1000
        )

        blue = st.selectbox(
            "Bluetooth",
            [0,1]
        )

        clock_speed = st.slider(
            "Clock Speed",
            0.5,
            3.5,
            2.0
        )

        dual_sim = st.selectbox(
            "Dual SIM",
            [0,1]
        )

        fc = st.slider(
            "Front Camera",
            0,
            20,
            5
        )

        four_g = st.selectbox(
            "4G",
            [0,1]
        )

        int_memory = st.slider(
            "Internal Memory",
            2,
            128,
            32
        )

        m_dep = st.slider(
            "Mobile Depth",
            0.1,
            1.0,
            0.5
        )

        mobile_wt = st.slider(
            "Weight",
            80,
            250,
            150
        )

        n_cores = st.slider(
            "CPU Cores",
            1,
            8,
            4
        )

    with col2:

        pc = st.slider(
            "Primary Camera",
            0,
            25,
            12
        )

        px_height = st.number_input(
            "Pixel Height",
            0,
            2000,
            1000
        )

        px_width = st.number_input(
            "Pixel Width",
            500,
            3000,
            1500
        )

        ram = st.number_input(
            "RAM",
            256,
            4000,
            2000
        )

        sc_h = st.slider(
            "Screen Height",
            5,
            20,
            12
        )

        sc_w = st.slider(
            "Screen Width",
            0,
            20,
            6
        )

        talk_time = st.slider(
            "Talk Time",
            2,
            20,
            10
        )

        three_g = st.selectbox(
            "3G",
            [0,1]
        )

        touch_screen = st.selectbox(
            "Touch Screen",
            [0,1]
        )

        wifi = st.selectbox(
            "WiFi",
            [0,1]
        )

    df = pd.DataFrame({

        "battery_power":[battery_power],
        "blue":[blue],
        "clock_speed":[clock_speed],
        "dual_sim":[dual_sim],
        "fc":[fc],
        "four_g":[four_g],
        "int_memory":[int_memory],
        "m_dep":[m_dep],
        "mobile_wt":[mobile_wt],
        "n_cores":[n_cores],
        "pc":[pc],
        "px_height":[px_height],
        "px_width":[px_width],
        "ram":[ram],
        "sc_h":[sc_h],
        "sc_w":[sc_w],
        "talk_time":[talk_time],
        "three_g":[three_g],
        "touch_screen":[touch_screen],
        "wifi":[wifi]

    })

    st.success("Manual input captured successfully.")

    ############
    # ======================================================
# DATASET SUMMARY
# ======================================================

st.markdown("---")

st.header("📊 Dataset Summary")

total_rows = len(df)
total_columns = len(df.columns)
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📄 Rows",
        value=f"{total_rows:,}"
    )

with col2:
    st.metric(
        label="📋 Columns",
        value=total_columns
    )

with col3:
    st.metric(
        label="❗ Missing Values",
        value=missing_values
    )

with col4:
    st.metric(
        label="📑 Duplicate Rows",
        value=duplicate_rows
    )

# ======================================================
# DATASET PREVIEW
# ======================================================

st.markdown("---")

st.header("👀 Dataset Preview")

# preview_rows = st.slider(
#     "Rows to display",
#     min_value=5,
#     max_value=min(100, len(df)),
#     value=min(10, len(df))
# )
if len(df) > 1:
    preview_rows = st.slider(
        "Rows to display",
        min_value=1,
        max_value=len(df),
        value=min(10, len(df))
    )
else:
    preview_rows = 1

st.dataframe(df.head(preview_rows), use_container_width=True)

st.dataframe(
    df.head(preview_rows),
    use_container_width=True
)

# ======================================================
# DATA TYPES
# ======================================================

with st.expander("📋 View Dataset Information"):

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

# ======================================================
# DESCRIPTIVE STATISTICS
# ======================================================

with st.expander("📈 Descriptive Statistics"):

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # ======================================================
# PREDICTION
# ======================================================

st.markdown("---")

st.header("🤖 Mobile Price Prediction")

if st.button("🚀 Predict Price Range", type="primary"):

    model = models[selected_model]

    # Scale only for Logistic Regression and KNN
    if selected_model in ["Logistic Regression", "KNN"]:

        X = scaler.transform(df)

    else:

        X = df.copy()

    predictions = model.predict(X)

    result = df.copy()

    result["Predicted Price Range"] = predictions

    # Convert numeric prediction into readable label

    labels = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }

    result["Price Category"] = result["Predicted Price Range"].map(labels)

    st.success("Prediction completed successfully!")

    st.markdown("---")

    st.subheader("📋 Prediction Results")

    st.dataframe(
        result,
        use_container_width=True
    )

    # ==========================================
    # Prediction Summary
    # ==========================================

    st.subheader("📊 Prediction Summary")

    summary = (
        result["Price Category"]
        .value_counts()
        .rename_axis("Price Category")
        .reset_index(name="Count")
    )

    col1, col2 = st.columns([1,1])

    with col1:

        st.dataframe(
            summary,
            use_container_width=True
        )

    with col2:

        st.bar_chart(
            summary.set_index("Price Category")
        )

    # ==========================================
    # Download CSV
    # ==========================================

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Prediction CSV",

        data=csv,

        file_name="mobile_price_predictions.csv",

        mime="text/csv"

    )

# ======================================================
# MODEL PERFORMANCE COMPARISON
# ======================================================

st.markdown("---")

st.header("🏆 Model Performance Comparison")

performance_file = "outputs/model_performance.csv"

if os.path.exists(performance_file):

    performance_df = pd.read_csv(performance_file)

    st.dataframe(
        performance_df,
        use_container_width=True
    )

    # ------------------------------------------
    # Best Model
    # ------------------------------------------

    best_model = performance_df.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]

    st.success(
        f"🥇 Best Performing Model: "
        f"**{best_model['Model']}** "
        f"(Accuracy: {best_model['Accuracy']:.4f})"
    )

    # ------------------------------------------
    # Accuracy Chart
    # ------------------------------------------

    st.subheader("📊 Accuracy Comparison")

    chart_df = performance_df.set_index("Model")

    st.bar_chart(chart_df["Accuracy"])

    # ------------------------------------------
    # Detailed Metrics
    # ------------------------------------------

    st.subheader("📈 Detailed Evaluation Metrics")

    metric = st.selectbox(
        "Select Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
            "AUC"
        ]
    )

    st.bar_chart(chart_df[metric])

else:

    st.warning(
        "model_performance.csv not found.\n"
        "Run train_models.ipynb first."
    )

# ==========================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# ==========================================================

st.markdown("---")
st.subheader("📈 Model Evaluation")

if "price_range" in df.columns:

    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        classification_report,
    )

    X_actual = df.drop(columns=["price_range"])

    if "id" in X_actual.columns:
        X_actual = X_actual.drop(columns=["id"])

    y_actual = df["price_range"]

    # Scale only for Logistic Regression and KNN
    if selected_model in ["Logistic Regression", "K-Nearest Neighbors"]:
        X_eval = scaler.transform(X_actual)
    else:
        X_eval = X_actual

    y_pred = model.predict(X_eval)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Confusion Matrix")

        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_actual,
            y_pred,
            cmap="Blues",
            ax=ax,
        )

        st.pyplot(fig)

    with col2:

        st.markdown("### Classification Report")

        report = classification_report(
            y_actual,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.style.format("{:.3f}"),
            use_container_width=True
        )

else:
    st.info(
        "Upload a dataset containing the 'price_range' column "
        "to display the Confusion Matrix and Classification Report."
    )

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.markdown("---")
st.subheader("🤖 Selected Model Information")

model_info = {
    "Logistic Regression":
        "A linear classification algorithm that works well on linearly separable data. "
        "Requires feature scaling.",

    "Decision Tree":
        "Tree-based algorithm that is easy to interpret and does not require feature scaling.",

    "K-Nearest Neighbors":
        "Predicts based on the nearest neighbours. Sensitive to feature scaling.",

    "Naive Bayes":
        "Probabilistic classifier based on Bayes' theorem with independence assumption.",

    "Random Forest":
        "An ensemble of Decision Trees that generally provides the best performance and reduces overfitting."
}

st.info(model_info[selected_model])

# ==========================================================
# DOWNLOAD TRAINED MODELS
# ==========================================================

st.markdown("---")
st.subheader("📦 Download Trained Models")

model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "K-Nearest Neighbors": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}

col1, col2 = st.columns(2)

with col1:

    if os.path.exists(model_files[selected_model]):
        with open(model_files[selected_model], "rb") as file:
            st.download_button(
                label=f"⬇ Download {selected_model} Model",
                data=file,
                file_name=os.path.basename(model_files[selected_model]),
                mime="application/octet-stream"
            )

with col2:

    if os.path.exists("models/scaler.pkl"):
        with open("models/scaler.pkl", "rb") as file:
            st.download_button(
                label="⬇ Download Scaler",
                data=file,
                file_name="scaler.pkl",
                mime="application/octet-stream"
            )

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.markdown("---")

with st.expander("ℹ Project Information"):

    st.markdown("""
### Machine Learning Assignment

**Algorithms Implemented**

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Random Forest

**Evaluation Metrics**

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

**Dataset**

Mobile Price Classification Dataset

**Deployment**

Developed using Streamlit.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;'>

### 🎓 BITS Pilani – Machine Learning Assignment

Developed using **Python**, **Scikit-learn** and **Streamlit**

</div>
""",
unsafe_allow_html=True,
)