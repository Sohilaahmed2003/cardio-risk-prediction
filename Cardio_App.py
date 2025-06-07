def patch_sklearn_tags():
    from sklearn.base          import BaseEstimator, TransformerMixin
    from sklearn.pipeline      import Pipeline
    from sklearn.compose       import ColumnTransformer
    from sklearn.preprocessing import ( OneHotEncoder, OrdinalEncoder, StandardScaler,FunctionTransformer)
    for cls in (BaseEstimator, TransformerMixin,Pipeline, ColumnTransformer,OneHotEncoder,OrdinalEncoder, StandardScaler, FunctionTransformer):
        if not hasattr(cls, "sklearn_tags"):
            cls.sklearn_tags = lambda self: {}

patch_sklearn_tags()


import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def yesno_to_int(val):
    return 1 if val == "yes" else 0

required_files = [
    r"c:\Users\sohila\Downloads\xgboost_best_model.pkl",
    r"c:\Users\sohila\Downloads\preprocessor.pkl",
    r"c:\Users\sohila\Downloads\feature_selector.pkl",
    r"c:\Users\sohila\Downloads\cleaned_data.csv"
]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"❌ Missing required file(s): {', '.join(missing_files)}. Please ensure all files are present in the app directory.")
    st.stop()

@st.cache_resource
def load_model_and_pipeline():
    with open(r"c:\Users\sohila\Downloads\xgboost_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(r"c:\Users\sohila\Downloads\preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open(r"c:\Users\sohila\Downloads\feature_selector.pkl", "rb") as f:
        selector = pickle.load(f)

    def patch(obj):
        cls = obj.__class__
        if not hasattr(cls, "sklearn_tags"):
            cls.sklearn_tags = lambda self: {}
        if isinstance(obj, Pipeline):
            for _, step in obj.steps:
                patch(step)
        if isinstance(obj, ColumnTransformer):
            for _, tr, _ in obj.transformers:
                patch(tr)

    patch(preprocessor)
    patch(selector)
    return model, preprocessor, selector

model, preprocessor, selector = load_model_and_pipeline()
@st.cache_data
def load_dataset():
    df = pd.read_csv(r"c:\Users\sohila\Downloads\cleaned_data.csv")
    if "cardio" in df.columns:
        df = df.drop(columns=["cardio"])  
    return df

df_viz = load_dataset()

# App Title
st.title("🫀 Cardiovascular Disease Risk Predictor")
st.markdown("Use the tabs below to explore predictions or visualize data.")

# Tabs
tab1, tab2 = st.tabs(["Predict Risk", "Dashboard"])

with tab1:
    st.markdown("### 🩺 Enter your health information:")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", options=["male", "female"], help="Select your biological sex.")
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, help="Enter your height in centimeters.")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, help="Enter your weight in kilograms.")
        ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=240, value=120, help="Upper number of your blood pressure.")
        ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=80, help="Lower number of your blood pressure.")
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], index=0,
                                   help="1: Normal, 2: Above Normal, 3: Well Above Normal")
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3], index=0,
                             help="1: Normal, 2: Above Normal, 3: Well Above Normal")

    with col2:
        smoke = st.selectbox("Do you smoke?", ["yes", "no"], help="Select 'yes' if you currently smoke.")
        alco = st.selectbox("Do you consume alcohol?", ["yes", "no"], help="Select 'yes' if you regularly consume alcohol.")
        active = st.selectbox("Are you physically active?", ["yes", "no"], help="Select 'yes' if you exercise regularly.")
        age_years = st.number_input("Age (years)", min_value=18, max_value=120, value=45, help="Enter your age in years.")
        lifestyle_score = st.slider("Lifestyle Score (0=Poor, 10=Excellent)", 0, 10, 5, help="Rate your overall lifestyle.")
        bp_category = st.selectbox("Blood Pressure Category", [
            "Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"
        ], help="Select your blood pressure category as diagnosed.")
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**BMI**: {bmi:.2f} ({bmi_category})")
    with col2:
        st.write(f"**Pulse Pressure**: {ap_hi - ap_lo}")

    with st.expander("See your input summary"):
        st.json({
            "Gender": gender,
            "Height (cm)": height,
            "Weight (kg)": weight,
            "Age (years)": age_years,
            "Systolic BP": ap_hi,
            "Diastolic BP": ap_lo,
            "Cholesterol": cholesterol,
            "Glucose": gluc,
            "Smoke": smoke,
            "Alcohol": alco,
            "Active": active,
            "Lifestyle Score": lifestyle_score,
            "BP Category": bp_category,
            "BMI": f"{bmi:.2f}",
            "BMI Category": bmi_category,
            "Pulse Pressure": ap_hi - ap_lo
        })

    if st.button("Predict Risk", type="primary", use_container_width=True):
        try:
            features_df = pd.DataFrame([{
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": yesno_to_int(smoke),
                "alco": yesno_to_int(alco),
                "active": yesno_to_int(active),
                "age_years": age_years,
                "bmi": bmi,
                "bp_category": bp_category,
                "pulse_pressure": ap_hi - ap_lo,
                "is_obese": 1 if bmi >= 30 else 0,
                "lifestyle_score": lifestyle_score,
                "bmi_category": bmi_category
            }])

            # Apply preprocessing
            X_preprocessed = preprocessor.transform(features_df)
            X_selected = selector.transform(X_preprocessed)

            prediction = model.predict(X_selected)[0]
            probability = model.predict_proba(X_selected)[0][1] * 100
            risk = "High Risk ⚠️" if prediction == 1 else "Low Risk ✅"

            st.markdown("---")
            st.subheader("📊 Prediction Result")
            st.markdown(f"### Risk Level: **{risk}**")
            st.markdown(f"### Probability of CVD: **{probability:.2f}%**")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Tab 2: Dashboard
with tab2:
    st.markdown("## 📊 Dashboard")
    # --- Overview Metrics ---
    num_rows, num_cols = df_viz.shape
    num_numerical = len(df_viz.select_dtypes(include=np.number).columns)
    num_categorical = len(df_viz.select_dtypes(include=['object', 'category']).columns)

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", f"{num_rows}")
    m2.metric("Numerical Features", f"{num_numerical}")
    m3.metric("Categorical Features", f"{num_categorical}")

    st.markdown("---")

    # --- Univariate Section ---
    st.subheader("🔹 Univariate Distributions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Numerical**")
        num_col = st.selectbox("Select a numerical column", df_viz.select_dtypes(include=np.number).columns, key="uni_num")
        fig, ax = plt.subplots()
        sns.histplot(df_viz[num_col], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**Categorical**")
        obj_cols = df_viz.select_dtypes(include='object').columns.tolist()
        cat_cols = df_viz.select_dtypes(include='category').columns.tolist()
        combined_cat_cols = obj_cols + cat_cols
        if combined_cat_cols:
            cat_col = st.selectbox("Select a categorical column", combined_cat_cols, key="uni_cat")
            fig, ax = plt.subplots()
            sns.countplot(data=df_viz, x=cat_col, ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No categorical variables found.")

    st.markdown("---")

    # --- Bivariate Section ---
    st.subheader("🔄 Bivariate Relationships")
    col1, col2, col3 = st.columns([3, 3, 4])

    x_var = col1.selectbox("X-axis", df_viz.columns, key="bi_x")
    y_var = col2.selectbox("Y-axis", df_viz.select_dtypes(include=np.number).columns, key="bi_y")
    hue = col3.selectbox("Color by (optional)", [None] + df_viz.columns.tolist(), key="bi_hue")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_viz, x=x_var, y=y_var, hue=hue, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.markdown("---")

    # --- Multivariate Section ---
    st.subheader("📈 Correlation Heatmap")
    selected_vars = st.multiselect(
        "Select numerical columns",
        df_viz.select_dtypes(include=np.number).columns.tolist(),
        default=df_viz.select_dtypes(include=np.number).columns.tolist()[:2]
    )

    if len(selected_vars) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(df_viz[selected_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Select two or more columns to generate a heatmap.")
