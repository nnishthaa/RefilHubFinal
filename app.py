
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
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ReFill Hub Intelligence", layout="wide")

df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80?text=ReFill+Hub")
    st.markdown("### ♻️ Sustainable Smart Refill Kiosk Platform")

    st.markdown("## 🌱 What do you want to see?")
    page = st.radio("", ["🏠 Dashboard Overview", "🧩 About ReFill Hub", "📊 Analysis"])

    st.markdown("---")
    st.markdown("### 👥 Team Members")
    st.write("👑 **Nishtha – Insights Lead**\nProvides deep consumer insights and data interpretations.")
    st.write("✨ **Anjali – Data Analyst**\nManages data cleaning and ensures analytical accuracy.")
    st.write("🌱 **Amatulla – Sustainability Research**\nAligns project objectives with eco‑friendly practices.")
    st.write("📊 **Amulya – Analytics Engineer**\nBuilds analytical models and evaluation frameworks.")
    st.write("🧠 **Anjan – Strategy & AI**\nLeads strategic direction using AI‑driven insights.")

# Dashboard Overview
if page == "🏠 Dashboard Overview":
    st.title("ReFill Hub – Eco Intelligence Dashboard")

    st.write("""
The ReFill Hub is a sustainability-focused retail solution deploying automated smart refill kiosks across the UAE for daily essentials such as shampoos, detergents, and cooking oils.  
The core mission is to minimize single-use plastic waste by encouraging refillable consumption supported by digital payments and frictionless user experience.

The business targets young professionals and eco-conscious urban residents.  
Survey insights confirm middle-income groups and sustainability-aware respondents as early adopters.  
Future vision includes expanding into non-liquid categories and scaling across the GCC.
""")

    c1, c2 = st.columns(2)
    c1.metric("Total Responses", df.shape[0])
    c2.metric("Total Features", df.shape[1])

# About Page
elif page == "🧩 About ReFill Hub":
    st.header("About ReFill Hub")
    st.write("""
ReFill Hub aims to eliminate plastic waste by replacing single-use packaging with automated refill kiosks.  
The platform connects sustainability with convenience through app-driven interactions and a strong focus on user-friendly experience.
""")

# ANALYSIS
elif page == "📊 Analysis":
    st.title("Analysis Dashboard")

    tab = st.tabs(["Classification", "Regression", "Clustering", "Association Rules", "Insights"])

    # Classification
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

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        metrics_table = []
        cols = st.columns(2)
        grid_index = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:,1]

            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Greens", ax=ax)
            ax.set_title(f"{name} – Confusion Matrix")
            cols[grid_index%2].pyplot(fig)
            grid_index += 1

            # ROC
            fig2, ax2 = plt.subplots(figsize=(4,3))
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax2.plot(fpr, tpr)
            ax2.set_title(f"{name} – ROC Curve")
            cols[grid_index%2].pyplot(fig2)
            grid_index += 1

            rep = classification_report(y_test, preds, output_dict=True)
            metrics_table.append([
                name,
                rep['weighted avg']['precision'],
                rep['weighted avg']['recall'],
                rep['weighted avg']['f1-score'],
                accuracy_score(y_test, preds)
            ])

        df_metrics = pd.DataFrame(metrics_table, columns=["Model","Precision","Recall","F1 Score","Accuracy"])
        st.subheader("📊 Model Comparison Table")
        st.dataframe(df_metrics)

    # Regression
    with tab[1]:
        st.subheader("Willingness-to-Pay Regression")

        df_r = df.dropna(subset=["Willingness_to_Pay_AED"])
        df_r2 = df_r.copy()

        for col in df_r2.select_dtypes(include=['object']).columns:
            df_r2[col] = le.fit_transform(df_r2[col])

        X = df_r2.drop(columns=["Willingness_to_Pay_AED"])
        y = df_r2["Willingness_to_Pay_AED"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = LinearRegression().fit(X_train, y_train)
        preds = reg.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        st.write(f"### MAE: {mae:.2f}")
        st.write(f"### RMSE: {rmse:.2f}")

        st.write("""
A lower MAE and RMSE indicate stronger pricing prediction accuracy.  
This helps identify suitable pricing ranges for refill options while understanding consumer affordability patterns.
""")

    # Clustering
    with tab[2]:
        st.subheader("Customer Clustering")
        df_cl = df.select_dtypes(include=['float64','int64'])

        kmeans = KMeans(n_clusters=3, random_state=42).fit(df_cl)
        df['Cluster'] = kmeans.labels_

        st.write("Cluster distribution:")
        st.bar_chart(df['Cluster'].value_counts())

    # Association Rules
    with tab[3]:
        st.subheader("Association Rule Mining")

        df_ar = df.copy()
        cat_cols = df_ar.select_dtypes(include=['object']).columns.tolist()

        for col in cat_cols:
            df_ar[col] = df_ar[col].astype(str)

        df_hot = pd.get_dummies(df_ar[cat_cols]).fillna(0)

        try:
            freq = apriori(df_hot, min_support=0.05, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1)
            st.dataframe(rules.sort_values("lift", ascending=False).head(10))
        except Exception as e:
            st.error(f"Association Rules could not run: {e}")

    # Insights
    with tab[4]:
        st.subheader("Insights")

        insights = [
            "Eco-aware users demonstrate higher refill adoption rates.",
            "Middle-income sustainability-focused consumers form the strongest target group.",
            "Awareness of plastic ban significantly boosts refill interest.",
            "Higher sustainability scores link to greater willingness to pay.",
            "Popular refill locations help determine optimal kiosk placement."
        ]

        for i in insights:
            st.markdown(f"✔️ {i}")


# === AUTO-MERGED ADDONS ===

# === Added Dashboard Overview Enhancements ===
st.markdown("<h1 style='background:linear-gradient(90deg,#6a11cb,#2575fc); padding:25px; border-radius:12px; color:white;'>♻️ ReFill Hub – Eco Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.write(" ")
st.markdown("""
<div style='background:#d8f5d0; padding:25px; border-radius:12px; color:#000000;'>
<h3>💡 ReFill Hub: Business Overview</h3>
<p>The ReFill Hub is a sustainability-focused retail solution deploying automated smart refill kiosks across the UAE for essentials...</p>
<p>The service targets urban dwellers and young professionals. Business model includes refill margins, partnerships, and subscriptions.</p>
<p>Long-term roadmap includes GCC expansion and non-liquid refills.</p>
</div>
""", unsafe_allow_html=True)

# === Tab Styling ===
st.markdown("""<style>
.stTabs [data-baseweb="tab"] {color:white !important; font-size:18px;}
.stTabs [aria-selected="true"] {color:#ff4d4d !important; border-bottom:3px solid #ff4d4d !important;}
</style>""", unsafe_allow_html=True)

# === About Page Cards ===
if page == "🧩 About ReFill Hub":
    st.title("About ReFill Hub")
    cols = st.columns(3)
    for i, card in enumerate([
        ("Mission","Driving refill culture across UAE."),
        ("Vision","Plastic-free future with smart kiosks."),
        ("Business Model","Refill margins + brand partnerships."),
        ("Target Audience","Eco-aware urban professionals."),
        ("Impact","Cuts single-use plastic dramatically."),
        ("Roadmap","GCC expansion + non-liquid categories.")
    ]):
        with cols[i%3]:
            st.markdown(f"""<div style='background:#f5f5f5; padding:20px; border-radius:10px;'>
            <h4>{card[0]}</h4><p>{card[1]}</p></div>""", unsafe_allow_html=True)

