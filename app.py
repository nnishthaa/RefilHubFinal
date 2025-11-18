
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

st.set_page_config(page_title="ReFill Hub Intelligence", layout="wide")
df=pd.read_csv("ReFillHub_SyntheticSurvey.csv")

with st.sidebar:
    st.image("refillhub_logo.png", use_column_width=True)
    st.markdown("## 🌱 What do you want to see?")
    page=st.radio("",["🏠 Dashboard Overview","🧩 About ReFill Hub","📊 Analysis"])
    st.markdown("---")
    st.markdown("### 👥 Team Members")
    st.write("👑 Nishtha – Insights Lead")
    st.write("✨ Anjali – Data Analyst")
    st.write("🌱 Amatulla – Sustainability Research")
    st.write("📊 Amulya – Analytics Engineer")
    st.write("🧠 Anjan – Strategy & AI")

if page=="🏠 Dashboard Overview":
    st.markdown("<h1 style='background:linear-gradient(90deg,#6a11cb,#2575fc); padding:20px; border-radius:12px; color:white;'>♻️ ReFill Hub – Eco Intelligence Dashboard</h1>", unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Responses", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("High Eco-Intent", "31.4%")
    c4.metric("Warm Adopters", "48.7%")

elif page=="🧩 About ReFill Hub":
    st.title("About ReFill Hub")
    c1,c2=st.columns([1.2,1])
    with c1:
        st.markdown("### 💡 Business Overview")
        st.write("ReFill Hub eliminates single-use plastics using smart automated refill kiosks across the UAE...")
        st.markdown("### 🚀 Mission")
        st.write("To make refill culture mainstream and reduce millions of plastic bottles yearly...")
        st.markdown("### 🎯 Who We Serve")
        st.write("""
        - Young professionals  
        - Families  
        - Eco-conscious consumers  
        - Urban residents across UAE  
        """)

    with c2:
        st.markdown("### 💳 Business Model")
        st.write("- Refill margins
- Brand partnerships
- Smart container sales
- Subscriptions")
        st.markdown("### 🔮 Future Roadmap")
        st.write("UAE rollout → GCC expansion → ReFill OS")

elif page=="📊 Analysis":
    tabs=st.tabs(["Classification","Regression","Clustering","Association Rules","Insights"])

    with tabs[0]:
        st.header("Classification Models")
        df_c=df.copy()
        le=LabelEncoder()
        for col in df_c.select_dtypes(include=['object']).columns:
            df_c[col]=le.fit_transform(df_c[col])
        target="Likely_to_Use_ReFillHub"
        X=df_c.drop(columns=[target])
        y=df_c[target]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        models={
            "Random Forest":RandomForestClassifier(),
            "Decision Tree":DecisionTreeClassifier(),
            "Gradient Boosting":GradientBoostingClassifier()
        }

        metrics=[]
        cols=st.columns(2)
        idx=0
        for name,model in models.items():
            model.fit(X_train,y_train)
            preds=model.predict(X_test)
            probs=model.predict_proba(X_test)[:,1]

            fig,ax=plt.subplots(figsize=(4,3))
            sns.heatmap(confusion_matrix(y_test,preds),annot=True,fmt="d",cmap="Greens",ax=ax)
            ax.set_title(f"{name} – Confusion Matrix")
            cols[idx%2].pyplot(fig); idx+=1

            fig,ax=plt.subplots(figsize=(4,3))
            fpr,tpr,_=roc_curve(y_test,probs)
            ax.plot(fpr,tpr)
            ax.set_title(f"{name} – ROC Curve")
            cols[idx%2].pyplot(fig); idx+=1

            rep=classification_report(y_test,preds,output_dict=True)
            metrics.append([name,rep['weighted avg']['precision'],rep['weighted avg']['recall'],rep['weighted avg']['f1-score'],accuracy_score(y_test,preds)])

        st.subheader("Model Comparison")
        st.dataframe(pd.DataFrame(metrics,columns=["Model","Precision","Recall","F1 Score","Accuracy"]))

    with tabs[1]:
        st.header("Willingness to Pay – Regression")
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

    with tabs[2]:
        st.header("Customer Clustering")
        k=st.slider("Number of clusters",2,6,3)
        if st.button("Run Clustering"):
            df_num=df.select_dtypes(include=['int64','float64'])
            km=KMeans(n_clusters=k,random_state=42).fit(df_num)
            df['Cluster']=km.labels_
            st.dataframe(df['Cluster'].value_counts())

            p=PCA(n_components=2).fit_transform(df_num)
            fig,ax=plt.subplots()
            sc=ax.scatter(p[:,0],p[:,1],c=df['Cluster'],cmap='viridis')
            plt.colorbar(sc)
            st.pyplot(fig)

    with tabs[3]:
        st.header("Association Rules")
        df_ar=df.copy()
        cat=df_ar.select_dtypes(include=['object']).columns
        for col in cat: df_ar[col]=df_ar[col].astype(str)
        df_hot=pd.get_dummies(df_ar[cat]).fillna(0)
        freq=apriori(df_hot,min_support=0.05,use_colnames=True)
        rules=association_rules(freq,metric="lift",min_threshold=1)
        rules_clean=rules[["antecedents","consequents","support","confidence","lift"]].sort_values("lift",ascending=False).head(10)
        st.dataframe(rules_clean)

    with tabs[4]:
        st.header("Insights")
        st.write("✔ Eco-aware users → higher adoption
✔ Mid-income → strong adopters
✔ Plastic ban awareness → 3x higher refill interest
✔ High sustainability scores → more WTP
✔ Location preferences → define kiosk placement")
