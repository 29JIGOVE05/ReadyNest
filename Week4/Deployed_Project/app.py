import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
 
st.set_page_config(page_title="Delhi Web-Dev Lead Finder", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "delhi_business_clean.csv")
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["price_level_clean"] = df["price_level_clean"].fillna(0)
    return df
 
@st.cache_resource
def train_model(df):
    features_num = ["review_count", "rating", "num_types", "price_level_clean"]
    features_cat = ["area", "category_searched"]
    X = df[features_num + features_cat]
    y = df["has_website"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ("num", StandardScaler(), features_num),
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=12,
                                          random_state=42, n_jobs=-1,
                                          class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
 
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
 
    return pipeline, acc, auc, features_num, features_cat
 
 
def build_leads(df, pipeline, features_num, features_cat):
    X_all = df[features_num + features_cat]
    df = df.copy()
    df["website_propensity_score"] = pipeline.predict_proba(X_all)[:, 1]
 
    no_site = df[df["has_website"] == 0].copy()
    no_site["lead_score"] = (
        no_site["website_propensity_score"] * 0.7
        + no_site["review_count"].rank(pct=True) * 0.3
    )
    return no_site.sort_values("lead_score", ascending=False)
 
# LOAD + TRAIN (cached, runs once per deploy)
df = load_data()
pipeline, acc, auc, features_num, features_cat = train_model(df)
leads = build_leads(df, pipeline, features_num, features_cat)
st.title("🌐 Delhi Web-Dev Lead Finder")
st.caption(
    "Finds businesses in Delhi with no website that look just like the "
    "businesses that already have one -- ranked as sales leads."
)
col1, col2, col3 = st.columns(3)
col1.metric("Businesses analyzed", f"{len(df):,}")
col2.metric("Model accuracy", f"{acc*100:.1f}%")
col3.metric("Model ROC-AUC", f"{auc:.2f}")
 
st.divider()
tab1, tab2 = st.tabs(["📋 Browse Leads", "🔮 Score a New Business"])
 
with tab1:
    st.subheader("Ranked leads (no website, high propensity)")
 
    c1, c2 = st.columns(2)
    area_filter = c1.multiselect("Filter by area", sorted(df["area"].dropna().unique()))
    cat_filter = c2.multiselect("Filter by category", sorted(df["category_searched"].dropna().unique()))
 
    filtered = leads.copy()
    if area_filter:
        filtered = filtered[filtered["area"].isin(area_filter)]
    if cat_filter:
        filtered = filtered[filtered["category_searched"].isin(cat_filter)]
 
    st.dataframe(
        filtered[["name", "area", "category_searched", "rating", "review_count",
                  "website_propensity_score", "lead_score"]].head(100),
        use_container_width=True,
        column_config={
            "website_propensity_score": st.column_config.ProgressColumn(
                "Website Propensity", min_value=0, max_value=1, format="%.2f"),
            "lead_score": st.column_config.ProgressColumn(
                "Lead Score", min_value=0, max_value=1, format="%.2f"),
        },
    )
    st.download_button(
        "Download filtered leads as CSV",
        filtered.to_csv(index=False),
        file_name="hot_leads.csv",
        mime="text/csv",
    )
 
with tab2:
    st.subheader("Would a business like this typically have a website?")
    st.caption("Enter a business's profile to see its website-propensity score.")
 
    c1, c2 = st.columns(2)
    area_in = c1.selectbox("Area", sorted(df["area"].dropna().unique()))
    cat_in = c2.selectbox("Category", sorted(df["category_searched"].dropna().unique()))
    rating_in = c1.slider("Rating", 1.0, 5.0, 4.2, 0.1)
    reviews_in = c2.number_input("Review count", min_value=0, value=50)
    types_in = c1.number_input("Number of listed business types", min_value=1, value=3)
    price_in = c2.selectbox("Price level", ["Unknown", "Inexpensive", "Moderate", "Expensive", "Very Expensive"])
    price_map = {"Unknown": 0, "Inexpensive": 1, "Moderate": 2, "Expensive": 3, "Very Expensive": 4}
 
    if st.button("Score this business", type="primary"):
        input_row = pd.DataFrame([{
            "review_count": reviews_in,
            "rating": rating_in,
            "num_types": types_in,
            "price_level_clean": price_map[price_in],
            "area": area_in,
            "category_searched": cat_in,
        }])
        score = pipeline.predict_proba(input_row)[0, 1]
        st.metric("Website Propensity Score", f"{score:.2f}")
        if score >= 0.7:
            st.success("🔥 Hot lead -- this business's profile strongly resembles businesses that already have a website.")
        elif score >= 0.4:
            st.info("🌤️ Moderate lead -- worth outreach, but not a top priority.")
        else:
            st.warning("❄️ Weak lead -- this profile doesn't strongly resemble businesses with websites.")
 
st.divider()
st.caption("Model: Random Forest classifier predicting `has_website` from area, category, "
           "rating, review count, listing completeness, and price tier.")
