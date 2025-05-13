import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer

# Load model, preprocessor, and feature selector
@st.cache_resource
def load_model_and_pipeline():
    with open(r"c:\Users\sohila\Documents\xgboost_best_model.json", "rb") as f:
        model = pickle.load(f)
    with open(r"c:\Users\sohila\Documents\preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open(r"c:\Users\sohila\Documents\feature_selector.pkl", "rb") as f:
        selector = pickle.load(f)
    return model, preprocessor, selector

model, preprocessor, selector = load_model_and_pipeline()

# Load dataset for visualization
@st.cache_data
def load_dataset():
    df = pd.read_csv(r"c:\Users\sohila\Documents\cleaned_data.csv")
    if "cardio" in df.columns:
        df = df.drop(columns=["cardio"])  # Drop target column
    return df

df_viz = load_dataset()

# App Title
st.title("🫀 Cardiovascular Disease Risk Predictor")
st.markdown("Use the tabs below to explore predictions or visualize data.")

# Tabs
tab1, tab2 = st.tabs(["Predict Risk", "Data Visualizations"])

with tab1:
    st.markdown("### 🩺 Enter your health information:")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", options=["male", "female"])
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
        ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=240, value=120)
        ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=80)
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], index=0,
                                   help="1: Normal, 2: Above Normal, 3: Well Above Normal")
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3], index=0,
                             help="1: Normal, 2: Above Normal, 3: Well Above Normal")

    with col2:
        smoke = st.selectbox("Do you smoke?", ["yes", "no"])
        alco = st.selectbox("Do you consume alcohol?", ["yes", "no"])
        active = st.selectbox("Are you physically active?", ["yes", "no"])
        age_years = st.number_input("Age (years)", min_value=18, max_value=120, value=45)
        lifestyle_score = 0  # Replace with real input if used
        bp_category = st.selectbox("Blood Pressure Category", [
            "Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"
        ])
        bmi_category = "Obese" if (weight / ((height / 100) ** 2) >= 30) else "Normal"

    # Derived features shown as info
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**BMI**: {weight / ((height / 100) ** 2):.2f}")
    with col2:
        st.write(f"**Pulse Pressure**: {ap_hi - ap_lo}")

    # Prediction Button
    if st.button("Predict Risk", type="primary", use_container_width=True):
        try:
            # Create DataFrame only when button is clicked
            features_df = pd.DataFrame([{
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active,
                "age_years": age_years,
                "bmi": weight / ((height / 100) ** 2),
                "bp_category": bp_category,
                "pulse_pressure": ap_hi - ap_lo,
                "is_obese": 1 if weight / ((height / 100) ** 2) >= 30 else 0,
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

# Tab 2: Data Visualizations
with tab2:
    st.markdown("### 📊 Data Visualizations")
    st.markdown("Explore univariate, bivariate, and multivariate analysis below.")

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

    with viz_tab1:
        st.markdown("#### 🔹 Numerical Variables Distribution")
        num_col = st.selectbox("Select numerical variable", df_viz.select_dtypes(include=np.number).columns.tolist(), key="univar_num")
        fig, ax = plt.subplots()
        sns.histplot(df_viz[num_col], kde=True, ax=ax)
        st.pyplot(fig)

        st.markdown("#### 🔸 Categorical Variables Distribution")

        obj_cols = df_viz.select_dtypes(include='object').columns.tolist()
        cat_cols = df_viz.select_dtypes(include='category').columns.tolist()
        combined_cat_cols = obj_cols + cat_cols if obj_cols or cat_cols else []

        cat_col = st.selectbox("Select categorical variable", combined_cat_cols, key="univar_cat") if combined_cat_cols else None
        if cat_col:
            fig, ax = plt.subplots()
            sns.countplot(data=df_viz, x=cat_col, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No categorical columns found in dataset.")

    with viz_tab2:
        st.markdown("#### 🔄 Bivariate Relationships")
        cols = df_viz.columns.tolist()
        x_var = st.selectbox("X-axis", cols, index=0, key="bi_x")
        y_var = st.selectbox("Y-axis", cols, index=1, key="bi_y")

        plot_type = st.radio("Plot Type", ["Scatter", "Bar", "Box", "Grouped Bar"], horizontal=True)

        if plot_type == "Scatter":
            if np.issubdtype(df_viz[x_var].dtype, np.number) and np.issubdtype(df_viz[y_var].dtype, np.number):
                selected_hue = st.selectbox("Color by (optional)", ["None"] + cols, key="bi_color")
                hue_param = selected_hue if selected_hue != "None" else None
                fig, ax = plt.subplots()
                sns.scatterplot(data=df_viz, x=x_var, y=y_var, hue=hue_param, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Scatter plot requires both variables to be numeric.")
        elif plot_type == "Bar":
            fig, ax = plt.subplots()
            sns.barplot(data=df_viz, x=x_var, y=y_var, ax=ax, ci=None)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        elif plot_type == "Box":
            fig, ax = plt.subplots()
            sns.boxplot(data=df_viz, x=x_var, y=y_var, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        elif plot_type == "Grouped Bar":
            grouped = df_viz.groupby(x_var)[y_var].value_counts().unstack(fill_value=0)
            fig, ax = plt.subplots()
            grouped.plot(kind="bar", stacked=False, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

    with viz_tab3:
        st.markdown("#### 📈 Multivariate Analysis")

        selected_vars = st.multiselect(
            "Select variables for correlation",
            df_viz.select_dtypes(include=np.number).columns.tolist(),
            default=df_viz.select_dtypes(include=np.number).columns.tolist()[:2]
        )
        if len(selected_vars) > 1:
            fig, ax = plt.subplots()
            sns.heatmap(df_viz[selected_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Please select at least two numerical variables for correlation matrix.")
