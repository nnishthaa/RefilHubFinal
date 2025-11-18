
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub Intelligence", layout="wide")

st.sidebar.title("ReFill Hub")

module = st.sidebar.radio("Navigate", ["Dashboard Overview","About ReFill Hub","Classification","Regression","Clustering","Association Rules","Insights"])

df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

if module=="Dashboard Overview":
    st.header("ReFill Hub – Smart Refill Kiosk Dashboard")
    st.write("""### Why This Dashboard Matters
- Understand eco‑consumer behaviour  
- Predict refill adoption  
- Support pricing & kiosk placement decisions  
""")

if module=="About ReFill Hub":
    st.header("Executive Summary")
    st.write("ReFill Hub introduces smart refill kiosks reducing plastic waste...")
    st.subheader("Team Members")
    st.write("👑 **Nishtha – Insights Lead**: Drives consumer insights.")
    st.write("✨ **Anjali – Data Analyst**: Ensures data accuracy.")
    st.write("🌱 **Amatulla – Sustainability Research**: Aligns ESG goals.")
    st.write("📊 **Amulya – Analytics Engineer**: Builds analytical models.")
    st.write("🧠 **Anjan – Strategy & AI**: Leads AI-driven strategy.")

if module=="Classification":
    st.header("Classification Models")
    df2=df.copy()
    le=LabelEncoder()
    for col in df2.select_dtypes(include=['object']).columns:
        df2[col]=le.fit_transform(df2[col])
    X=df2.drop('Likely_to_Use_ReFillHub',axis=1)
    y=df2['Likely_to_Use_ReFillHub']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestClassifier().fit(X_train,y_train)
    preds=model.predict(X_test)
    probs=model.predict_proba(y_test.index)[:,1] if hasattr(model,'predict_proba') else np.zeros(len(y_test))
    st.text(classification_report(y_test,preds))
    fpr,tpr,_=roc_curve(y_test,probs)
    fig,ax=plt.subplots()
    ax.plot(fpr,tpr)
    ax.set_title("ROC Curve")
    st.pyplot(fig)

if module=="Regression":
    st.header("Willingness-to-Pay Regression")
    df2=df.copy()
    le=LabelEncoder()
    for col in df2.select_dtypes(include=['object']).columns:
        df2[col]=le.fit_transform(df2[col])
    X=df2.drop('Willingness_to_Pay_AED',axis=1)
    y=df2['Willingness_to_Pay_AED']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    reg=LinearRegression().fit(X_train,y_train)
    preds=reg.predict(X_test)
    mae=mean_absolute_error(y_test,preds)
    rmse=np.sqrt(mean_squared_error(y_test,preds))
    st.write(f"### MAE: {mae}")
    st.write(f"### RMSE: {rmse}")
    st.write("Predicting eco‑consumer payment behaviour helps refine pricing...")

if module=="Clustering":
    st.header("Customer Clustering")
    st.write("Clustering placeholder.")

if module=="Association Rules":
    st.header("Association Rule Mining")
    st.write("Patterns in refill preferences.")
    # placeholder

if module=="Insights":
    st.header("Key Insights")
    st.write("- Eco‑friendly users show higher refill likelihood.")
