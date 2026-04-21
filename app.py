import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Netflix Analytics Pro", page_icon="🍿", layout="wide")

# --- 2. PREMIUM CSS (Hero + Glassmorphism Styling) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #000000; }
    .hero-text {
        background: linear-gradient(to right, #ffffff 0%, #E50914 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important; font-weight: 800 !important; letter-spacing: -2px !important; margin-bottom: 0px;
    }
    .hero-sub { color: #888; font-size: 1.2rem; margin-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px); padding: 25px; border-radius: 20px; transition: all 0.3s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px); border-color: #E50914;
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.25);
    }
    [data-testid="stMetricLabel"] { color: #E50914 !important; font-weight: 700; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: white !important; font-size: 2.8rem !important; font-weight: 800 !important; }
    .insight-card {
        background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px; border-radius: 15px; margin-bottom: 15px; line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 40px; }
    .stTabs [aria-selected="true"] { color: #E50914 !important; border-bottom-color: #E50914 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. OPTIMIZED DATA LOADING ---
@st.cache_data
def load_data():
    path = os.path.join("01_Dataset", "netflix_titles.csv")
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df['description'] = df['description'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("NR")
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors='coerce')
    df["release_year"] = pd.to_numeric(df["release_year"], errors='coerce').fillna(0).astype(int)
    df["add_month"] = df["date_added"].dt.month_name()
    df["add_year"] = df["date_added"].dt.year
    df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df["genres_list"] = df["listed_in"].str.split(", ")
    df["clean_title"] = df["title"].str.strip()
    return df

@st.cache_resource
def compute_similarity(_df):
    tfidf = TfidfVectorizer(stop_words='english')
    content_soup = _df['listed_in'] + " " + _df['description']
    tfidf_matrix = tfidf.fit_transform(content_soup)
    return linear_kernel(tfidf_matrix, tfidf_matrix)

df = load_data()
if df is not None:
    cosine_sim = compute_similarity(df)

# --- 4. SIDEBAR FILTERS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=140)
st.sidebar.markdown("### FILTERS")

content_type = st.sidebar.multiselect("TYPE", options=df["type"].unique(), default=df["type"].unique())
year_range = st.sidebar.slider("RELEASE YEAR", int(df["release_year"].min()), int(df["release_year"].max()), (2015, 2024))
all_countries = sorted(list(set(df["country"].unique())))
selected_countries = st.sidebar.multiselect("NATIONS", options=all_countries)

st.sidebar.divider()
# --- ACTOR & DIRECTOR AUTOCOMPLETE RESTORED ---
director_list = sorted(list(set([d for d in df["director"].unique() if d != "Unknown"])))
actor_list = sorted(list(set([a.strip() for sublist in df["cast"].dropna().str.split(",") for a in sublist if a.strip() != "Unknown"])))

search_director = st.sidebar.selectbox("DIRECTOR", options=director_list, index=None, placeholder="Search...", accept_new_options=True)
search_cast = st.sidebar.selectbox("ACTOR", options=actor_list, index=None, placeholder="Search...", accept_new_options=True)

# --- 5. FILTERING LOGIC ---
mask = (df["type"].isin(content_type)) & (df["release_year"].between(year_range[0], year_range[1]))
if selected_countries: mask &= (df["country"].isin(selected_countries))
if search_director: mask &= df["director"].str.contains(search_director, case=False)
if search_cast: mask &= df["cast"].str.contains(search_cast, case=False)
filtered_df = df[mask]

# --- 6. HERO SECTION ---
st.markdown('<p class="hero-text">NETFLIX CATALOG ANALYTICS</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI-Powered Content Intelligence & Smart Recommendations</p>', unsafe_allow_html=True)

if not filtered_df.empty:
    # Bento Insights
    top_genre = pd.Series([g for sublist in filtered_df["genres_list"] for g in sublist]).mode()[0]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<div class="insight-card"><h5 style="color:#E50914;margin:0;">TOP GENRE</h5><b>{top_genre}</b> leads this segment.</div>', unsafe_allow_html=True)
    with col_b:
        avg_rt = filtered_df[filtered_df["type"]=="Movie"]["duration_num"].mean()
        st.markdown(f'<div class="insight-card"><h5 style="color:#E50914;margin:0;">VIEWER STATS</h5>Avg Movie Runtime: <b>{avg_rt:.0f} min</b></div>', unsafe_allow_html=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Titles", f"{len(filtered_df):,}")
    m2.metric("Nations", filtered_df["country"].nunique())
    m3.metric("Avg Year", int(filtered_df["release_year"].mean()))
    m4.metric("Ratings", filtered_df["rating"].nunique())

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 DASHBOARD", "🌍 GEOGRAPHY", "🔬 DEEP DIVE", "📋 CATALOG", "🤖 SMART RECS"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Media Mix")
            fig_pie = px.pie(filtered_df, names='type', hole=0.6, color_discrete_sequence=['#E50914', '#333333'])
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("Release Trend")
            growth = filtered_df.groupby("release_year").size().reset_index(name='count')
            st.plotly_chart(px.line(growth, x='release_year', y='count', color_discrete_sequence=['#E50914']).update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white"), use_container_width=True)

    with tab2:
        st.subheader("Production Centers (Top 10)")
        top_c = filtered_df["country"].value_counts().head(10).reset_index()
        fig_bar = px.bar(top_c, x='count', y='country', orientation='h', color_discrete_sequence=['#E50914'])
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Content Themes (Word Cloud)")
        text = " ".join(filtered_df["description"].dropna())
        if text:
            wordcloud = WordCloud(width=1200, height=400, background_color='black', colormap='Reds').generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(15, 5), facecolor='black'); ax_wc.imshow(wordcloud); plt.axis("off")
            st.pyplot(fig_wc)
        
        st.divider()
        st.subheader("Interactive Hierarchy")
        filtered_df['primary_genre'] = filtered_df['genres_list'].str[0]
        fig_sun = px.sunburst(filtered_df, path=['type', 'rating', 'primary_genre'], color='type', color_discrete_map={'Movie': '#E50914', 'TV Show': '#333333'})
        fig_sun.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}')
        fig_sun.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig_sun, use_container_width=True)

        st.divider()
        # --- MONTHLY HEATMAP RESTORED ---
        st.subheader("Addition Trends (Year vs Month)")
        heat_data = filtered_df.groupby(['add_year', 'add_month']).size().unstack(fill_value=0)
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        heat_data = heat_data.reindex(columns=[m for m in months if m in heat_data.columns])
        if not heat_data.empty:
            fig_h, ax_h = plt.subplots(figsize=(14, 7), facecolor='black')
            sns.heatmap(heat_data, cmap="YlOrRd", annot=True, fmt="d", ax=ax_h, annot_kws={"weight": "bold"})
            plt.setp(ax_h.get_xticklabels(), color="white"); plt.setp(ax_h.get_yticklabels(), color="white")
            ax_h.set_facecolor('black')
            st.pyplot(fig_h)

    with tab4:
        st.subheader("Data Explorer")
        st.dataframe(filtered_df[["title", "type", "director", "cast", "country", "release_year", "rating", "duration"]], use_container_width=True)
        st.download_button("📥 EXPORT CSV", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name="netflix_data.csv")

    with tab5:
        st.header("🤖 ML Smart Recommendations")
        pick = st.selectbox("Select a Title:", sorted(df['clean_title'].unique()))
        if pick:
            idx = df[df['clean_title'] == pick].index[0]
            scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
            recs = df.iloc[[i[0] for i in scores]][['title', 'type', 'listed_in', 'rating']]
            st.table(recs)
else:
    st.warning("No matches found.")