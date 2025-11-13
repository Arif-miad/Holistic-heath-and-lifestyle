# holistic_health_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import plotly.graph_objects as go

# =========================
# App Title
# =========================
st.set_page_config(page_title="Holistic Health Dashboard", layout="wide")
st.title("🌿 Holistic Health & Lifestyle Dashboard")

# =========================
# Dataset Upload
# =========================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())
else:
    st.warning("Upload your Holistic Health CSV dataset to start!")
    st.stop()

# =========================
# Tabs for EDA, Regression, Classification, Dashboard
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Regression", "Classification", "Interactive Dashboard"])

# =========================
# Tab 1: EDA
# =========================
with tab1:
    st.header("Exploratory Data Analysis")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
    
    st.subheader("Distribution Plots")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    
    st.subheader("Health Status Counts")
    if 'Health_Status' in df.columns:
        sns.countplot(x='Health_Status', data=df)
        plt.title("Health Status Distribution")
        st.pyplot(plt)

# =========================
# Tab 2: Regression
# =========================
with tab2:
    st.header("Regression: Predict Overall Health Score")
    
    # Features and target
    X = df[['Physical_Activity','Nutrition_Score','Stress_Level','Mindfulness','Sleep_Hours',
            'Hydration','BMI','Alcohol','Smoking']]
    y = df['Overall_Health_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.subheader("Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    st.write("RMSE:", round(mean_squared_error(y_test, y_pred_lr, squared=False),2))
    st.write("R2 Score:", round(r2_score(y_test, y_pred_lr),2))
    
    st.subheader("Random Forest Regression")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    st.write("RMSE:", round(mean_squared_error(y_test, y_pred_rf, squared=False),2))
    st.write("R2 Score:", round(r2_score(y_test, y_pred_rf),2))

# =========================
# Tab 3: Classification
# =========================
with tab3:
    st.header("Classification: Predict Health Status")
    
    if 'Health_Status' not in df.columns:
        st.warning("Health_Status column not found!")
    else:
        y_class = df['Health_Status']
        X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred_class = clf.predict(X_test)
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred_class))
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_class), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Feature Importance")
        importances = clf.feature_importances_
        feat_names = X.columns
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=importances, y=feat_names, ax=ax)
        ax.set_title("Feature Importance for Health Status")
        st.pyplot(fig)

# =========================
# Tab 4: Interactive Dashboard
# =========================
with tab4:
    st.header("Interactive Health Dashboard")
    st.write("Input your lifestyle metrics to see predicted Health Score and Status.")
    
    # User Inputs
    pa = st.slider("Physical Activity (minutes/day)", 0, 120, 45)
    nutrition = st.slider("Nutrition Score (0-10)", 0, 10, 7)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    mindfulness = st.slider("Mindfulness (minutes/day)", 0, 60, 15)
    sleep = st.slider("Sleep Hours", 3, 10, 7)
    hydration = st.slider("Hydration (liters/day)", 0.5, 5.0, 2.5)
    bmi = st.number_input("BMI", 18.0, 40.0, 24.0)
    alcohol = st.slider("Alcohol units/week", 0, 20, 3)
    smoking = st.slider("Cigarettes/day", 0, 30, 5)
    
    # Create feature array
    input_features = np.array([[pa,nutrition,stress,mindfulness,sleep,hydration,bmi,alcohol,smoking]])
    
    # Scale input for regression
    input_scaled = scaler.transform(input_features)
    
    # Predict Health Score and Status
    predicted_score = lr.predict(input_scaled)[0]
    predicted_status = clf.predict(input_features)[0]
    
    st.metric("Predicted Overall Health Score", round(predicted_score,2))
    st.metric("Predicted Health Status", predicted_status)
    
    # Radar chart for visualization
    categories = ['Physical Activity','Nutrition','Stress','Mindfulness','Sleep','Hydration','BMI','Alcohol','Smoking']
    values = [pa, nutrition*12, 10-stress, mindfulness, sleep*6, hydration*12, 40-bmi, 20-alcohol, 30-smoking]
    
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    st.plotly_chart(fig)
