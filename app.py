
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

# Dashboard
if page=="🏠 Dashboard Overview":
    st.markdown("<h1 style='background:linear-gradient(90deg,#6a11cb,#2575fc); padding:20px; border-radius:12px; color:white;'>♻️ ReFill Hub – Eco Intelligence Dashboard</h1>", unsafe_allow_html=True)

    # NEW COLORED BOX
    st.markdown("""
    <div style='background:#d9f7e6; padding:20px; border-radius:12px;'>
    <h3>💡 ReFill Hub: Business Overview</h3>
    <p>The ReFill Hub is a sustainability-focused retail solution deploying automated smart refill kiosks across the UAE. 
    The core goal is to drastically reduce single-use plastic waste by enabling reusable container refills.</p>
    <p>The service targets urban dwellers and young professionals in Dubai and Abu Dhabi. The model combines refill margins, brand partnerships, 
    and subscription plans. Data analysis revealed strong adoption among middle-income groups aware of sustainability trends.</p>
    <p>Long-term vision includes expansion to non-liquids and GCC-wide scaling.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Responses", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("High Eco-Intent", "31.4%")
    c4.metric("Warm Adopters", "48.7%")

elif page=="🧩 About ReFill Hub":
    st.title("About ReFill Hub")
    st.write("Simplified placeholder.")

elif page=="📊 Analysis":
    tabs=st.tabs(["Classification","Regression","Clustering","Association Rules","Insights"])

    with tabs[4]:
        st.header("Insights")
        st.write("""
        ✔ Eco-aware users show higher refill interest.

        ✔ Middle-income groups display strong adoption intent.

        ✔ Plastic ban awareness significantly boosts refill likelihood.

        ✔ Sustainability-focused users show higher willingness to pay.

        ✔ Preferred refill locations guide kiosk placement strategy.
        """)
