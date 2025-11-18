
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub Intelligence", layout="wide")

df=pd.read_csv("ReFillHub_SyntheticSurvey.csv")

with st.sidebar:
    st.image("refillhub_logo.png", use_column_width=True)
    st.markdown("## 🌱 What do you want to see?")
    page=st.radio("",["🏠 Dashboard Overview","🧩 About ReFill Hub","📊 Analysis"])
    st.markdown("### 👥 Team Members")
    st.write("👑 Nishtha – Insights Lead")
    st.write("✨ Anjali – Data Analyst")
    st.write("🌱 Amatulla – Sustainability Research")
    st.write("📊 Amulya – Analytics Engineer")
    st.write("🧠 Anjan – Strategy & AI")

# dashboard overview
if page=="🏠 Dashboard Overview":
    st.markdown("<h1 style='background: linear-gradient(90deg,#5f3dc4,#4dabf7); padding:20px; border-radius:15px; color:white;'>♻️ ReFill Hub – Eco Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.write("AI-powered cockpit that transforms survey data into refill adoption intelligence.")
    c1,c2,c3=st.columns(3)
    c1.metric("Total Responses",df.shape[0])
    c2.metric("Features Tracked",df.shape[1])
    c3.metric("Eco Personas","3+")

elif page=="🧩 About ReFill Hub":
    st.header("About ReFill Hub")
    st.write("""ReFill Hub eliminates plastic waste by deploying refill kiosks across the UAE...""")

elif page=="📊 Analysis":
    tabs=st.tabs(["Classification","Regression","Clustering","Association Rules","Insights"])

    # clustering
    with tabs[2]:
        st.subheader("Customer Clustering")
        k=st.slider("Number of clusters",2,6,3)
        if st.button("🔍 Run clustering"):
            df_num=df.select_dtypes(include=['int64','float64']).copy()
            km=KMeans(n_clusters=k, random_state=42).fit(df_num)
            df['Cluster']=km.labels_
            st.write("Cluster sizes:")
            st.dataframe(df['Cluster'].value_counts())

            # PCA plot
            p=PCA(n_components=2).fit_transform(df_num)
            fig,ax=plt.subplots()
            sc=ax.scatter(p[:,0],p[:,1],c=df['Cluster'],cmap='viridis')
            plt.colorbar(sc)
            st.pyplot(fig)

    # placeholder for others
    with tabs[0]: st.write("Classification models here...")
    with tabs[1]: st.write("Regression models here...")
    with tabs[3]: st.write("Association rules here...")
    with tabs[4]: st.write("Insights here...")

