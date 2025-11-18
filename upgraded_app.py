
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub Intelligence", layout="wide")
df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

# Sidebar
with st.sidebar:
    try:
        st.image("refillhub_logo.png", use_column_width=True)
    except:
        st.write("ReFill Hub")
    st.markdown("## 🌱 What do you want to see?")
    page = st.radio("", ["🏠 Dashboard Overview", "🧩 About ReFill Hub", "📊 Analysis"])
    st.markdown("---")
    st.markdown("### 👥 Team Members")
    st.write("👑 **Nishtha – Insights Lead**")
    st.write("✨ **Anjali – Data Analyst**")
    st.write("🌱 **Amatulla – Sustainability Research**")
    st.write("📊 **Amulya – Analytics Engineer**")
    st.write("🧠 **Anjan – Strategy & AI**")

# Dashboard Overview
if page == "🏠 Dashboard Overview":
    st.markdown("<h1
    # New boxes
    st.write(" ")
    st.markdown("""
    <div style='background:#d8f5d0; padding:25px; border-radius:12px; color:#000000;'>
    <h3>💡 ReFill Hub: Business Overview</h3>
    <p>The ReFill Hub is a sustainability-focused retail solution deploying automated smart refill kiosks across the UAE.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Responses", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("Eco Personas", "3+")
 style='background:linear-gradient(90deg,#6a11cb,#2575fc); padding:20px; border-radius:12px; color:white;'>♻️ ReFill Hub – Eco Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.write("""This dashboard transforms ReFill Hub’s survey data into actionable intelligence...""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Responses", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("Eco Personas", "3+")

elif page == "🧩 About ReFill Hub":
    st.header("About ReFill Hub")
    st.write("""ReFill Hub reduces plastic waste by deploying smart refill kiosks...""")

elif page == "📊 Analysis":
    tabs = st.tabs(["Classification", "Regression", "Clustering", "Association Rules", "Insights"])

    # Classification
    with tabs[0]:
        st.subheader("Classification Models")
        df_c = df.copy()
        le = LabelEncoder()
        for col in df_c.select_dtypes(include=['object']).columns:
            df_c[col] = le.fit_transform(df_c[col])
        target="Likely_to_Use_ReFillHub"
        X=df_c.drop(columns=[target])
        y=df_c[target]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        metrics=[]
        cols=st.columns(2)
        idx=0

        for name,model in models.items():
            model.fit(X_train,y_train)
            preds=model.predict(X_test)
            probs=model.predict_proba(X_test)[:,1]

            # Confusion matrix
            fig,ax=plt.subplots(figsize=(4,3))
            sns.heatmap(confusion_matrix(y_test,preds),annot=True,fmt="d",cmap="Greens",ax=ax)
            ax.set_title(f"{name} – Confusion Matrix")
            cols[idx%2].pyplot(fig)
            idx+=1

            # ROC
            fig2,ax2=plt.subplots(figsize=(4,3))
            fpr,tpr,_=roc_curve(y_test,probs)
            ax2.plot(fpr,tpr)
            ax2.set_title(f"{name} – ROC Curve")
            cols[idx%2].pyplot(fig2)
            idx+=1

            rep=classification_report(y_test,preds,output_dict=True)
            metrics.append([name, rep["weighted avg"]["precision"], rep["weighted avg"]["recall"], rep["weighted avg"]["f1-score"], accuracy_score(y_test,preds)])

        st.subheader("Model Comparison Table")
        st.dataframe(pd.DataFrame(metrics,columns=["Model","Precision","Recall","F1","Accuracy"]))

    # Regression
    with tabs[1]:
        st.subheader("Willingness-To-Pay Regression")
        df_r=df.dropna(subset=["Willingness_to_Pay_AED"])
        df_r2=df_r.copy()
        for col in df_r2.select_dtypes(include=['object']).columns:
            df_r2[col]=le.fit_transform(df_r2[col])
        X=df_r2.drop(columns=["Willingness_to_Pay_AED"])
        y=df_r2["Willingness_to_Pay_AED"]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        reg=LinearRegression().fit(X_train,y_train)
        preds=reg.predict(X_test)
        st.write("MAE:",mean_absolute_error(y_test,preds))
        st.write("RMSE:",np.sqrt(mean_squared_error(y_test,preds)))

    # Clustering (Premium)
    with tabs[2]:
        st.subheader("Customer Clustering")
        k=st.slider("Number of clusters",2,6,3)
        if st.button("🔍 Run Clustering"):
            df_num=df.select_dtypes(include=['float64','int64']).copy()
            km=KMeans(n_clusters=k,random_state=42).fit(df_num)
            df['Cluster']=km.labels_
            st.dataframe(df['Cluster'].value_counts())

            pca=PCA(n_components=2).fit_transform(df_num)
            fig,ax=plt.subplots()
            sc=ax.scatter(pca[:,0],pca[:,1],c=df['Cluster'],cmap='viridis')
            plt.colorbar(sc)
            st.pyplot(fig)

    # Association Rules
    with tabs[3]:
        st.subheader("Association Rule Mining")
        df_ar=df.copy()
        cat_cols=df_ar.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df_ar[col]=df_ar[col].astype(str)
        df_hot=pd.get_dummies(df_ar[cat_cols]).fillna(0)
        freq=apriori(df_hot,min_support=0.05,use_colnames=True)
        rules=association_rules(freq,metric="lift",min_threshold=1)
        st.dataframe(rules.sort_values("lift",ascending=False).head(10))

    # Insights
    with tabs[4]:
        st.subheader("Insights")
        insights=[
            "Eco-aware users show higher refill adoption.",
            "Middle-income consumers are strong early adopters.",
            "Plastic ban awareness boosts refill interest.",
            "Sustainability scores link to higher WTP.",
            "Refill locations guide better kiosk placement."
        ]
        for i in insights:
            st.markdown(f"✔️ {i}")
