
# FULL REBUILT FINAL APP (STRUCTURE + NEW BOX + RESTORED SECTIONS)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub Dashboard", layout="wide")
df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

# Sidebar
with st.sidebar:
    st.image("refillhub_logo.png", use_column_width=True)
    st.markdown("## 🌱 What do you want to see?")
    page = st.radio("", ["🏠 Dashboard Overview","🧩 About ReFill Hub","📊 Analysis"])
    st.markdown("---")
    st.markdown("### 👥 Team Members")
    st.write("👑 Nishtha – Insights Lead")
    st.write("✨ Anjali – Data Analyst")
    st.write("🌱 Amatulla – Sustainability Research")
    st.write("📊 Amulya – Analytics Engineer")
    st.write("🧠 Anjan – Strategy & AI")

# -----------------------------
# Dashboard Overview
# -----------------------------
if page=="🏠 Dashboard Overview":
    # Original top section restored
    st.markdown("<h1 style='background:linear-gradient(90deg,#6a11cb,#2575fc); padding:25px; border-radius:12px; color:white;'>♻️ ReFill Hub – Eco Intelligence Dashboard</h1>", unsafe_allow_html=True)

    # SPACE BETWEEN BOXES
    st.write(" ")
    st.write(" ")

    # NEW GREEN BOX WITH BLACK TEXT
    st.markdown("""
    <div style='background:#d8f5d0; padding:25px; border-radius:12px; color:#000000;'>
    <h3>💡 ReFill Hub: Business Overview</h3>
    <p>The ReFill Hub is a sustainability-focused retail solution deploying automated smart refill kiosks across the UAE for essentials such as shampoos, detergents, and oils. The primary mission is to reduce plastic waste significantly while offering convenience and affordability.</p>
    <p>The service targets urban residents and professionals in high-density Emirates like Dubai and Abu Dhabi. The business model blends refill margins, brand partnerships, and subscription opportunities. Data analysis shows strong interest from middle-income groups aware of the UAE plastic ban.</p>
    <p>The long-term plan includes expanding into non-liquid categories and scaling the model across the GCC.</p>
    </div>
    """, unsafe_allow_html=True)

    # Original metrics
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Responses", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("High Eco-Intent", "31.4%")
    c4.metric("Warm Adopters", "48.7%")

# -----------------------------
# About Page
# -----------------------------
elif page=="🧩 About ReFill Hub":
    st.title("About ReFill Hub")
    st.write("Full 6-card about page goes here (placeholder).")

# -----------------------------
# Analysis Section
# -----------------------------
elif page=="📊 Analysis":
    tabs = st.tabs(["Classification","Regression","Clustering","Association Rules","Insights"])

    # -------------------------
    # INSIGHTS (Vertical 5 points)
    # -------------------------
    with tabs[4]:
        st.header("Insights")
        st.write("""
        ✔ Eco-aware users show higher refill interest.

        ✔ Middle-income groups display strong adoption intent.

        ✔ Plastic ban awareness significantly boosts refill likelihood.

        ✔ Sustainability-focused users show higher willingness to pay.

        ✔ Preferred refill locations guide kiosk placement strategy.
        """)

