import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
from scipy.stats import linregress

BASE_DIR = Path(__file__).parent
st.set_page_config(
    page_title="Movie Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    h1 {
        color: #1f77b4;
        font-weight: 600;
        margin-bottom: 10px;
    }
    h2 {
        color: #2c3e50;
        font-weight: 500;
        margin-top: 20px;
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "tmdb_cleaned.csv")
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    try:
        model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
        with open(BASE_DIR / "models" / "feature_columns.json", "r") as f:
            features = json.load(f)
        with open(BASE_DIR / "models" / "model_metrics.json", "r") as f:
            metrics = json.load(f)
        return model, features, metrics
    except Exception:
        return None, None, None

# Load resources
df = load_data()
model, feature_cols, model_metrics = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section",["Dashboard", "Analysis", "Genres", "Financial", "Predictions", "Model Info"],index=0)
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Overview")
st.sidebar.info(f"""
**Movies**: {len(df):,}  
**Years**: {int(df['Release_Year'].min())} - {int(df['Release_Year'].max())}  
**Avg Rating**: {df['vote_average'].mean():.2f}  
**Genres**: {len([c for c in df.columns if c.startswith('Genre_')])}
""")

# Helper function
def get_genre_stats():
    genre_cols = [col for col in df.columns if col.startswith('Genre_')]
    stats = []
    for col in genre_cols:
        genre_name = col.replace('Genre_', '')
        genre_df = df[df[col] == 1]
        if len(genre_df) > 0:
            stats.append({'Genre': genre_name,'Count': len(genre_df),'Avg_Rating': genre_df['vote_average'].mean(),'Avg_Budget': genre_df['budget'].mean()})
    return pd.DataFrame(stats)

# DASHBOARD
if page == "Dashboard":
    st.title("Movie Analytics Dashboard")
    st.markdown("*Comprehensive insights from The Movie Database*")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies",f"{len(df):,}",f"{len(df[df['Release_Year'] >= 2020]):,} since 2020")
    
    with col2:
        st.metric("Average Rating",f"{df['vote_average'].mean():.2f}",f"σ={df['vote_average'].std():.2f}")
    
    with col3:
        budget_pct = (df['budget'] > 0).sum() / len(df) * 100
        st.metric("With Budget Data",f"{budget_pct:.1f}%",f"{(df['budget'] > 0).sum():,} movies")
    
    with col4:
        avg_runtime = df['runtime'].mean()
        st.metric("Average Runtime",f"{avg_runtime:.0f} min","Standard feature length")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating Distribution")
        fig = px.histogram(df,x='vote_average',nbins=40,title=None,labels={'vote_average': 'Rating', 'count': 'Frequency'},color_discrete_sequence=['#1f77b4'])
        fig.add_vline(x=df['vote_average'].mean(),line_dash="dash",line_color="red",annotation_text=f"Mean: {df['vote_average'].mean():.2f}")
        fig.update_layout(height=400, showlegend=False, hovermode='x')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Production Volume")
        yearly = df.groupby('Release_Year').size().reset_index(name='count')
        yearly = yearly[yearly['Release_Year'] >= 1990]
        fig = px.line(yearly,x='Release_Year',y='count',title=None,labels={'Release_Year': 'Year', 'count': 'Movies'},markers=True)
        fig.update_traces(line_color='#2ca02c', line_width=2)
        fig.update_layout(height=400, hovermode='x')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top movies
    st.subheader("Top Rated Movies (Min 100 votes)")
    top_movies = df[df['vote_count'] >= 100].nlargest(10, 'vote_average')[['title', 'vote_average', 'vote_count', 'Release_Year', 'runtime']].reset_index(drop=True)
    st.dataframe(top_movies.style.format({'vote_average': '{:.2f}','vote_count': '{:,.0f}','Release_Year': '{:.0f}','runtime': '{:.0f}'}),use_container_width=True,hide_index=True)

# ANALYSIS
elif page == "Analysis":
    st.title("Exploratory Analysis")
    
    # Correlation
    st.subheader("Feature Correlations")
    numeric_cols = ['vote_average', 'vote_count', 'popularity', 'runtime','Log_Budget', 'Log_Revenue', 'Log_Popularity', 'Log_Vote_Count']
    if 'ROI' in df.columns:
        numeric_cols.append('ROI')
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix,text_auto='.2f',aspect="auto",color_continuous_scale='RdBu_r',zmin=-1,zmax=1,title=None)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    
    # Budget vs Rating
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Budget vs Rating")
        budget_df = df[df['budget'] > 0].copy()
        fig = px.scatter(budget_df,x='Log_Budget',y='vote_average',color='Log_Popularity',hover_data=['title', 'Release_Year'],title=None,labels={'Log_Budget': 'Log(Budget)', 'vote_average': 'Rating'},color_continuous_scale='Viridis')
        
        if len(budget_df) > 1:
            slope, intercept, r_value, _, _ = linregress(budget_df['Log_Budget'],budget_df['vote_average'])
            x_range = np.linspace(budget_df['Log_Budget'].min(),budget_df['Log_Budget'].max(), 100)
            y_range = slope * x_range + intercept
            fig.add_trace(go.Scatter(x=x_range, y=y_range,mode='lines',name=f'Trend (R²={r_value**2:.3f})',line=dict(color='red', dash='dash')))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Runtime vs Rating")
        fig = px.scatter(df,x='runtime',y='vote_average',color='Log_Vote_Count',hover_data=['title', 'Release_Year'],title=None,labels={'runtime': 'Runtime (min)', 'vote_average': 'Rating'},color_continuous_scale='Plasma')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trends Over Time")
    yearly_stats = df.groupby('Release_Year').agg({'vote_average': ['mean', 'median'],'id': 'count'}).reset_index()
    yearly_stats.columns = ['Year', 'Mean', 'Median', 'Count']
    yearly_stats = yearly_stats[(yearly_stats['Year'] >= 1990) & (yearly_stats['Year'] <= 2024)]
    fig = make_subplots(rows=2, cols=1,subplot_titles=('Rating Trends', 'Movie Count'),vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Mean'],name='Mean', line=dict(color='blue', width=2)),row=1, col=1)
    fig.add_trace(go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Median'],name='Median', line=dict(color='green', width=2, dash='dash')),row=1, col=1)
    
    fig.add_trace(go.Bar(x=yearly_stats['Year'], y=yearly_stats['Count'],name='Count', marker_color='steelblue'),row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Rating", row=1, col=1)
    fig.update_yaxes(title_text="Movies", row=2, col=1)
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

# GENRES
elif page == "Genres":
    st.title("Genre Analysis")
    
    genre_stats = get_genre_stats()
    sort_by = st.selectbox("Sort by", ["Avg_Rating", "Count", "Avg_Budget"])
    genre_stats = genre_stats.sort_values(sort_by, ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Rating by Genre")
        fig = px.bar(
            genre_stats,
            y='Genre',
            x='Avg_Rating',
            orientation='h',
            title=None,
            color='Avg_Rating',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Number of Movies by Genre")
        fig = px.bar(
            genre_stats,
            y='Genre',
            x='Count',
            orientation='h',
            title=None,
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.subheader("Genre Statistics")
    display_stats = genre_stats.copy()
    display_stats['Avg_Budget'] = display_stats['Avg_Budget'].apply(lambda x: f"${x/1e6:.1f}M" if x > 0 else "N/A")
    st.dataframe(display_stats.style.format({'Avg_Rating': '{:.2f}', 'Count': '{:,.0f}'}),use_container_width=True,hide_index=True)

# FINANCIAL
elif page == "Financial":
    st.title("Financial Performance")
    if 'ROI' not in df.columns or df['ROI'].isna().all():
        st.warning("ROI data not available in dataset")
    else:
        roi_df = df[(df['ROI'].notna()) & (df['ROI'] > 0) & (df['ROI'] < 20)]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Median ROI", f"{roi_df['ROI'].median():.2f}x")
        with col2:
            profitable = (df['ROI'] > 1).sum()
            st.metric("Profitable", f"{profitable:,}")
        with col3:
            avg_budget = df[df['budget'] > 0]['budget'].mean()
            st.metric("Avg Budget", f"${avg_budget/1e6:.1f}M")
        with col4:
            avg_revenue = df[df['revenue'] > 0]['revenue'].mean()
            st.metric("Avg Revenue", f"${avg_revenue/1e6:.1f}M")
        st.markdown("---")
        
        # ROI Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROI Distribution")
            fig = px.histogram(roi_df, x='ROI', nbins=50, title=None)
            fig.add_vline(x=roi_df['ROI'].median(), line_dash="dash", annotation_text=f"Median: {roi_df['ROI'].median():.2f}x")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("ROI vs Rating")
            fig = px.scatter(roi_df, x='ROI', y='vote_average',color='Log_Budget', title=None)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top 15 Movies by ROI")
        top_roi = roi_df.nlargest(15, 'ROI')[['title', 'ROI', 'budget', 'revenue', 'vote_average']].copy()
        top_roi['budget'] = top_roi['budget'].apply(lambda x: f"${x/1e6:.1f}M")
        top_roi['revenue'] = top_roi['revenue'].apply(lambda x: f"${x/1e6:.1f}M")
        
        st.dataframe(top_roi.style.format({'ROI': '{:.2f}x', 'vote_average': '{:.2f}'}),use_container_width=True,hide_index=True)

# PREDICTIONS
elif page == "Predictions":
    st.title("Movie Rating Prediction")
    if model is None:
        st.error("Model files not found. Please train the model first.")
        st.stop()
    st.markdown("Enter movie details below to predict its rating")
    col1, col2 = st.columns(2)
    with col1:
        runtime = st.slider("Runtime (minutes)", 30, 300, 120)
        release_year = st.slider("Release Year", 1990, 2024, 2023)
        budget = st.number_input("Budget ($)", 0, 300000000, 50000000, step=1000000)
        popularity = st.slider("Popularity Score", 0.0, 100.0, 20.0)
    
    with col2:
        st.markdown("#### Select Genres")
        vote_count = st.slider("Expected Vote Count", 20, 10000, 500)
        genre_cols = [col for col in df.columns if col.startswith('Genre_')]
        selected_genres = {}
        
        for i, genre_col in enumerate(genre_cols):
            if i % 2 == 0:
                col_a, col_b = st.columns(2)
            if i % 2 == 0:
                with col_a:
                    selected_genres[genre_col] = st.checkbox(genre_col.replace('Genre_', ''))
            else:
                with col_b:
                    selected_genres[genre_col] = st.checkbox(genre_col.replace('Genre_', ''))
    
    if st.button("Predict Rating", type="primary", use_container_width=True):
        input_data = {'runtime': runtime,'Release_Year': release_year,'Has_Budget': 1 if budget > 0 else 0,'Log_Budget': np.log1p(budget),'Log_Popularity': np.log1p(popularity),'Log_Vote_Count': np.log1p(vote_count)}
        for genre_col in genre_cols:
            input_data[genre_col] = 1 if selected_genres.get(genre_col, False) else 0
        input_df = pd.DataFrame([input_data])[feature_cols]
        prediction = model.predict(input_df)[0]
        
        # Results
        st.markdown("---")
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Rating", f"{prediction:.2f}/10")
        
        with col2:
            if prediction >= 7.5:
                status = "Excellent"
            elif prediction >= 6.5:
                status = "Good"
            elif prediction >= 5.5:
                status = "Average"
            else:
                status = "Below Average"
            st.metric("Category", status)
        
        with col3:
            percentile = (df['vote_average'] < prediction).sum() / len(df) * 100
            st.metric("Better Than", f"{percentile:.1f}%")

        st.markdown("---")
        st.subheader("Similar Movies")
        similar = df.copy()
        similar['distance'] = ((similar['runtime'] - runtime).abs() / 100 +(similar['Release_Year'] - release_year).abs() / 10 +(similar['Log_Budget'] - np.log1p(budget)).abs())
        similar_movies = similar.nsmallest(5, 'distance')[['title', 'vote_average', 'runtime', 'Release_Year']].reset_index(drop=True)
        st.dataframe(similar_movies.style.format({'vote_average': '{:.2f}', 'runtime': '{:.0f}'}),use_container_width=True,hide_index=True)

# MODEL INFO
elif page == "Model Info":
    st.title("Model Performance")
    if model_metrics is None:
        st.error("Model metrics not found")
        st.stop()
    st.subheader("Model Comparison")
    metrics_df = pd.DataFrame(model_metrics['all_models'])
    st.dataframe(
        metrics_df.style.format({'rmse': '{:.4f}','mae': '{:.4f}','r2': '{:.4f}'}).background_gradient(subset=['r2'], cmap='RdYlGn'),use_container_width=True,hide_index=True
    )
    st.markdown("---")
    st.subheader("Best Model")
    best = model_metrics['best_model']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", best['model'])
    with col2:
        st.metric("RMSE", f"{best['rmse']:.4f}")
    with col3:
        st.metric("R² Score", f"{best['r2']:.4f}")
    with col4:
        st.metric("MAE", f"{best['mae']:.4f}")
        
    st.markdown("---")
    st.subheader("Model Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics_df['model'],y=metrics_df['r2'],name='R² Score',marker_color='lightblue'))
    fig.update_layout(xaxis_title="Model",yaxis_title="R² Score",height=400,showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    try:
        importance_df = pd.read_csv(BASE_DIR / "models" / "feature_importance.csv")
        if 'feature' not in importance_df.columns:
            importance_df.columns = ['feature', 'importance']
        st.markdown("---")
        st.subheader("Feature Importance")
        top_n = st.slider("Number of features", 5, len(importance_df), 10)
        top_features = importance_df.nlargest(top_n, 'importance')
        fig = px.bar(top_features,y='feature',x='importance',orientation='h',title=None,color='importance',color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception:
        pass

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 20px; font-size: 12px;'>
    Movie Analytics Dashboard
</div>
""", unsafe_allow_html=True)
