
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub Intelligence", layout="wide")

# Load dataset
df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/180x80?text=ReFill+Hub+Logo")
    st.markdown("### ♻️ Sustainable Smart Refill Kiosk Platform")
    st.markdown("## 🌱 What do you want to see?")
    page = st.radio("", ["🏠 Dashboard Overview", "🧩 About ReFill Hub", "📊 Analysis"])
    st.markdown("---")
    st.markdown("### 👥 Team Members")
    st.markdown("**👑 Nishtha – Insights Lead**")
    st.markdown("*Responsible for analysing consumer behaviour and generating insights.*")
    st.markdown("**✨ Anjali – Data Analyst**")
    st.markdown("*Ensures clean, structured and accurate datasets for analysis.*")
    st.markdown("**🌱 Amatulla – Sustainability Research**")
    st.markdown("*Aligns project goals with ESG and eco‑friendly outcomes.*")
    st.markdown("**📊 Amulya – Analytics Engineer**")
    st.markdown("*Builds data models and analytical systems.*")
    st.markdown("**🧠 Anjan – Strategy & AI**")
    st.markdown("*Leads strategic decision‑making through AI‑driven methods.*")

# Dashboard Overview
if page == "🏠 Dashboard Overview":
    st.title("ReFill Hub: Business Overview")
    st.write("""The ReFill Hub is a sustainability-focused retail solution deploying automated smart refill kiosks across the UAE for daily essentials such as shampoos, detergents, and cooking oils. The goal is to significantly reduce single-use plastic waste by enabling refillable consumption.

The service caters to urban residents and young professionals in major Emirates like Dubai and Abu Dhabi, supported by high mobile adoption and cashless technology. Revenue streams include per‑refill margins, FMCG brand partnerships, and future subscription models.

Survey analysis confirms strong viability, with middle‑income eco‑aware consumers being primary early adopters. The long‑term vision includes expanding into non‑liquids and scaling across the GCC.""")

    st.metric("Total Responses", df.shape[0])
    st.metric("Total Features", df.shape[1])

# About Page
elif page == "🧩 About ReFill Hub":
    st.header("About ReFill Hub")
    st.write("ReFill Hub aims to reduce plastic waste by replacing single‑use bottles with refillable dispensing kiosks. The platform integrates digital payments, eco‑friendly choices, and consumer convenience.")
    st.subheader("Executive Summary")
    st.write("Extensive survey data supports the platform’s market need, highlighting sustainability‑driven young professionals as key adopters.")

# Analysis Page
elif page == "📊 Analysis":
    st.title("Analysis Dashboard")
    tab = st.tabs(["Classification", "Regression", "Clustering", "Association Rules", "Insights"])

    # ---------------- Classification ----------------
    with tab[0]:
        st.subheader("Classification Models")
        df_c = df.copy()
        le = LabelEncoder()
        for col in df_c.select_dtypes(include=['object']).columns:
            df_c[col] = le.fit_transform(df_c[col])
        target = "Likely_to_Use_ReFillHub"
        X = df_c.drop(columns=[target])
        y = df_c[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {"Random Forest": RandomForestClassifier()}
        metrics_list = []

        cols = st.columns(2)
        idx = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:,1]

            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(4,3))
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
            ax.set_title(f"{name} – Confusion Matrix")
            cols[idx%2].pyplot(fig)
            idx+=1

            # ROC Curve
            fig2, ax2 = plt.subplots(figsize=(4,3))
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax2.plot(fpr, tpr)
            ax2.set_title(f"{name} – ROC Curve")
            cols[idx%2].pyplot(fig2)
            idx+=1

            report = classification_report(y_test, preds, output_dict=True)
            metrics_list.append([name, report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']])

        st.subheader("📊 Model Comparison Table")
        df_metrics = pd.DataFrame(metrics_list, columns=["Model","Precision","Recall","F1 Score"])
        st.dataframe(df_metrics)

    # ---------------- Regression ----------------
    with tab[1]:
        st.subheader("Willingness‑to‑Pay Regression")
        df_r = df.dropna(subset=["Willingness_to_Pay_AED"])
        df_r2 = df_r.copy()
        for col in df_r2.select_dtypes(include=['object']).columns:
            df_r2[col] = le.fit_transform(df_r2[col])
        X = df_r2.drop(columns=["Willingness_to_Pay_AED"])
        y = df_r2["Willingness_to_Pay_AED"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.write(f"### MAE: {mae:.2f}")
        st.write(f"### RMSE: {rmse:.2f}")
        st.write("WTP prediction helps in designing optimal pricing strategies, identifying high‑value eco‑consumers and understanding affordability patterns.")

    # ---------------- Clustering ----------------
    with tab[2]:
        st.subheader("Customer Clustering")
        df_cl = df.select_dtypes(include=['float64','int64'])
        kmeans = KMeans(n_clusters=3, random_state=42).fit(df_cl)
        df['Cluster'] = kmeans.labels_
        st.write("Cluster distribution:")
        st.bar_chart(df['Cluster'].value_counts())

    # ---------------- Association Rules ----------------
    with tab[3]:
        st.subheader("Association Rule Mining")
        df_ar = df.copy()
        for col in df_ar.select_dtypes(include='object').columns:
            df_ar[col] = df_ar[col].astype(str)
        df_hot = pd.get_dummies(df_ar)
        freq = apriori(df_hot, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1)
        st.dataframe(rules.sort_values("lift", ascending=False).head(10))

    # ---------------- Insights ----------------
    with tab[4]:
        st.subheader("Insights")
        insights = [
            "Eco-aware users show significantly higher refill adoption.",
            "Middle-income consumers are strong early adopters.",
            "Plastic ban awareness correlates with refill interest.",
            "Higher sustainability scores indicate higher willingness to pay.",
            "Preferred refill locations help identify future kiosk placement zones."
        ]
        for i in insights:
            st.markdown(f"✔️ {i}")
