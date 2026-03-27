import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        st.warning(f"NLTK download warning: {e}")

download_nltk_data()

st.set_page_config(page_title="GATEWAYS-2025 Dashboard", layout="wide", page_icon="�")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #fff8e1 0%, #fff3c4 100%); }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        border-radius: 15px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #ff6b35;
        margin-bottom: 0.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.1);
    }
    .metric-card:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #ff6b35; }
    .metric-label { font-size: 0.85rem; color: #e65100; text-transform: uppercase; letter-spacing: 1px; }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #2e7d32;
        border-bottom: 3px solid #ff6b35; padding-bottom: 6px; margin-bottom: 1rem;
    }
    .welcome-card {
        background: linear-gradient(135deg, #ffd54f 0%, #ffb74d 50%, #ff8a65 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
        animation: float 3s ease-in-out infinite;
    }
    .welcome-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .welcome-subtitle { font-size: 1.1rem; opacity: 0.9; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); }
    .filter-section {
        background: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.1);
    }
    .chart-container {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.1);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff3e0 0%, #ffe0b2 100%);
        border-right: 3px solid #ff6b35;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-radius: 8px 8px 0 0;
        color: #e65100;
        border: 2px solid #ffcc80;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #ffcc80, #ffb74d);
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b35, #e65100) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #fff3e0;
    }
    ::-webkit-scrollbar-thumb {
        background: #ff6b35;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #e65100;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("C5-FestDataset-fest_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/sun.png", width=80)
st.sidebar.title("🎛️ GATEWAYS-2025")
st.sidebar.markdown("---")

# Filter Section
with st.sidebar.expander("🔍 Data Filters", expanded=True):
    st.markdown("**Customize your view:**")
    
    all_events = ["All"] + sorted(df["Event Name"].unique().tolist())
    all_states = ["All"] + sorted(df["State"].unique().tolist())
    all_types  = ["All"] + sorted(df["Event Type"].unique().tolist())

    sel_event = st.selectbox("🎯 Filter by Event", all_events, help="Select specific event or view all")
    sel_state = st.selectbox("📍 Filter by State", all_states, help="Filter participants by state")
    sel_type  = st.selectbox("👥 Filter by Type", all_types, help="Individual or Group events")

# Apply Filters
dff = df.copy()
if sel_event != "All": dff = dff[dff["Event Name"] == sel_event]
if sel_state != "All": dff = dff[dff["State"] == sel_state]
if sel_type  != "All": dff = dff[dff["Event Type"] == sel_type]

# Filter Results
st.sidebar.markdown("---")
with st.sidebar.container():
    st.metric("📊 Filtered Results", len(dff), help=f"Showing {len(dff)} of {len(df)} total participants")
    
    # Quick insights
    if len(dff) > 0:
        avg_rating = dff['Rating'].mean()
        st.metric("⭐ Avg Rating", f"{avg_rating:.1f}", 
                 delta=f"{avg_rating - df['Rating'].mean():+.2f}" if len(df) > 0 else None)

# Export Section
with st.sidebar.expander("💾 Export Data", expanded=False):
    st.markdown("**Download filtered data:**")
    csv = dff.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="gateways_filtered_data.csv",
        mime="text/csv",
        help="Download current filtered dataset"
    )

# ── Header ───────────────────────────────────────────────────────────────────
# Welcome Section
st.markdown("""
<div class="welcome-card">
    <div class="welcome-title">🎓 GATEWAYS-2025</div>
    <div class="welcome-subtitle">National-Level Technical Fest · Analytics & Insights Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Quick Stats Overview
with st.expander("📈 Quick Overview", expanded=True):
    st.markdown("## 🎯 Key Performance Indicators")
    st.markdown("*Real-time insights from participant data*")
    st.markdown("---")

    # ── KPI Row ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Total Participants", len(dff)),
        ("Colleges",           dff["College"].nunique()),
        ("States Covered",     dff["State"].nunique()),
        ("Events",             dff["Event Name"].nunique()),
        ("Avg Rating",         f"{dff['Rating'].mean():.2f} ★"),
    ]
    for col, (label, val) in zip([k1,k2,k3,k4,k5], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 Participation Trends", "🗺️ India State Map", "💬 Feedback & Ratings", "📋 Data Explorer"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Participation Trends
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">📈 Participation Trends & Analytics</div>', unsafe_allow_html=True)
    
    # Progress indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        total_participants = len(dff)
        max_possible = len(df)
        progress = min(total_participants / max_possible, 1.0) if max_possible > 0 else 0
        st.metric("Participation Rate", f"{progress:.1%}")
        st.progress(progress)
    
    with col2:
        unique_colleges = dff["College"].nunique()
        max_colleges = df["College"].nunique()
        college_progress = unique_colleges / max_colleges if max_colleges > 0 else 0
        st.metric("College Coverage", f"{unique_colleges}/{max_colleges}")
        st.progress(college_progress)
    
    with col3:
        unique_states = dff["State"].nunique()
        max_states = df["State"].nunique()
        state_progress = unique_states / max_states if max_states > 0 else 0
        st.metric("State Coverage", f"{unique_states}/{max_states}")
        st.progress(state_progress)

    st.markdown("---")
    
    with st.container():
        st.markdown("### Event & College Participation")
        c1, c2 = st.columns(2)

        # Event-wise bar
        with c1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            ev_cnt = dff["Event Name"].value_counts().reset_index()
            ev_cnt.columns = ["Event", "Participants"]
            fig = px.bar(ev_cnt, x="Participants", y="Event", orientation="h",
                         color="Participants", color_continuous_scale="Sunset",
                         title="Participants per Event")
            fig.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", font_color="#2e7d32",
                              coloraxis_showscale=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # College-wise horizontal bar (top 10)
        with c2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            col_cnt = dff["College"].value_counts().head(10).reset_index()
            col_cnt.columns = ["College", "Participants"]
            fig2 = px.bar(col_cnt, x="Participants", y="College", orientation="h",
                          color="Participants", color_continuous_scale="Viridis",
                          title="Top 10 Colleges by Participation")
            fig2.update_layout(showlegend=False, plot_bgcolor="white",
                               paper_bgcolor="white", font_color="#2e7d32",
                               coloraxis_showscale=False, height=380)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown("### Event Types & Revenue Analysis")
        c3, c4 = st.columns(2)

        # Individual vs Group donut
        with c3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            type_cnt = dff["Event Type"].value_counts().reset_index()
            type_cnt.columns = ["Type", "Count"]
            fig3 = px.pie(type_cnt, names="Type", values="Count", hole=0.55,
                          color_discrete_sequence=["#4da6ff","#f77f00"],
                          title="Individual vs Group Events")
            fig3.update_layout(paper_bgcolor="#0f1117", font_color="#e0e6f0", height=340)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Revenue per event
        with c4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            rev = dff.groupby("Event Name")["Amount Paid"].sum().reset_index()
            rev.columns = ["Event", "Revenue (₹)"]
            fig4 = px.bar(rev, x="Event", y="Revenue (₹)",
                          color="Revenue (₹)", color_continuous_scale="Sunset",
                          title="Revenue Generated per Event (₹)")
            fig4.update_layout(showlegend=False, plot_bgcolor="white",
                               paper_bgcolor="white", font_color="#2e7d32",
                               coloraxis_showscale=False, height=340,
                               xaxis_tickangle=-30)
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Heatmap: college × event
    with st.container():
        st.markdown('<div class="section-title">🔥 College × Event Participation Heatmap</div>', unsafe_allow_html=True)
        top_colleges = dff["College"].value_counts().head(10).index.tolist()
        heat_df = dff[dff["College"].isin(top_colleges)]
        pivot = heat_df.groupby(["College","Event Name"]).size().unstack(fill_value=0)
        fig5 = px.imshow(pivot, color_continuous_scale="RdYlGn",
                         title="Participation Intensity (Top 10 Colleges)",
                         aspect="auto")
        fig5.update_layout(paper_bgcolor="white", font_color="#2e7d32", height=420)
        st.plotly_chart(fig5, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — India Map
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">🗺️ Geographic Distribution & State Analysis</div>', unsafe_allow_html=True)
    
    # Map Controls
    with st.expander("🗺️ Map Controls", expanded=False):
        map_style = st.selectbox("Map Style:", ["Choropleth", "Bubble Map"], index=0)
        color_scheme = st.selectbox("Color Scheme:", ["Blues", "Reds", "Greens", "Purples"], index=0)
    
    with st.container():
        st.markdown("### India State Participation Map")
        state_cnt = dff["State"].value_counts().reset_index()
        state_cnt.columns = ["State", "Participants"]

        # Plotly choropleth using India GeoJSON
        geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
        import requests, json
        resp = requests.get(geojson_url)
        india_geo = resp.json()

        fig_map = px.choropleth(
            state_cnt,
            geojson=india_geo,
            featureidkey="properties.ST_NM",
            locations="State",
            color="Participants",
            color_continuous_scale=color_scheme.lower(),
            title="Number of Participants by State",
            hover_data={"Participants": True}
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            paper_bgcolor="white", font_color="#2e7d32",
            geo_bgcolor="white", height=580,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with st.container():
        st.markdown("### State-wise Participation Rankings")
        fig_state_bar = px.bar(
            state_cnt.sort_values("Participants", ascending=True),
            x="Participants", y="State", orientation="h",
            color="Participants", color_continuous_scale=color_scheme.lower(),
            title="Participants per State"
        )
        fig_state_bar.update_layout(
            showlegend=False, plot_bgcolor="white",
            paper_bgcolor="white", font_color="#2e7d32",
            coloraxis_showscale=False, height=420
        )
        st.plotly_chart(fig_state_bar, use_container_width=True)
        
        # State statistics
        top_state = state_cnt.loc[state_cnt["Participants"].idxmax()]
        st.metric(f"🏆 Top State: {top_state['State']}", top_state["Participants"], "participants")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Feedback & Ratings
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Feedback & Rating Analysis</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)

    # Rating distribution
    with f1:
        rat_cnt = dff["Rating"].value_counts().sort_index().reset_index()
        rat_cnt.columns = ["Rating", "Count"]
        fig_r = px.bar(rat_cnt, x="Rating", y="Count",
                       color="Count", color_continuous_scale="RdYlGn",
                       title="Rating Distribution (1–5)")
        fig_r.update_layout(showlegend=False, plot_bgcolor="white",
                             paper_bgcolor="white", font_color="#2e7d32",
                             coloraxis_showscale=False, height=360)
        st.plotly_chart(fig_r, use_container_width=True)

    # Avg rating per event
    with f2:
        avg_rat = dff.groupby("Event Name")["Rating"].mean().reset_index()
        avg_rat.columns = ["Event", "Avg Rating"]
        avg_rat = avg_rat.sort_values("Avg Rating", ascending=False)
        fig_ar = px.bar(avg_rat, x="Event", y="Avg Rating",
                        color="Avg Rating", color_continuous_scale="RdYlGn",
                        range_color=[1,5], title="Average Rating per Event")
        fig_ar.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", font_color="#2e7d32",
                              coloraxis_showscale=False, height=360,
                              xaxis_tickangle=-30)
        st.plotly_chart(fig_ar, use_container_width=True)

    # Feedback keyword frequency (word cloud data as bar)
    st.markdown('<div class="section-title">Most Common Feedback Keywords</div>', unsafe_allow_html=True)
    
    # NLTK-based text processing
    @st.cache_data
    def process_feedback_text(feedback_series):
        # Initialize NLTK tools
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # Additional custom stopwords for fest context
        custom_stopwords = {
            'fest', 'event', 'college', 'student', 'participants', 'good', 'great', 
            'excellent', 'amazing', 'wonderful', 'nice', 'best', 'really', 'very',
            'much', 'well', 'also', 'get', 'got', 'one', 'two', 'three', 'four', 'five'
        }
        stop_words.update(custom_stopwords)
        
        all_words = []
        
        for feedback in feedback_series.dropna():
            if isinstance(feedback, str):
                # Convert to lowercase
                text = feedback.lower()
                
                # Remove special characters and numbers
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                
                # Tokenize
                tokens = word_tokenize(text)
                
                # Remove stopwords and short words, lemmatize
                processed_tokens = [
                    lemmatizer.lemmatize(word) for word in tokens
                    if word not in stop_words and len(word) > 3
                ]
                
                all_words.extend(processed_tokens)
        
        return all_words
    
    # Process feedback text
    feedback_words = process_feedback_text(dff["Feedback on Fest"])
    
    # Get word frequencies
    word_freq = Counter(feedback_words)
    top_words = pd.DataFrame(word_freq.most_common(20), columns=["Word","Count"])
    
    fig_wf = px.bar(top_words, x="Count", y="Word", orientation="h",
                    color="Count", color_continuous_scale="Bluered",
                    title="Top 20 Feedback Keywords (NLTK Processed)")
    fig_wf.update_layout(showlegend=False, plot_bgcolor="white",
                         paper_bgcolor="white", font_color="#2e7d32",
                         coloraxis_showscale=False, height=500)
    st.plotly_chart(fig_wf, use_container_width=True)
    
    # Add word cloud option
    if st.checkbox("Show Word Cloud", value=False):
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            # Generate word cloud
            wordcloud_text = ' '.join(feedback_words)
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(wordcloud_text)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            st.pyplot(fig)
            
        except ImportError:
            st.info("Install wordcloud package for word cloud visualization: pip install wordcloud matplotlib")

    # Sentiment Analysis
    with st.container():
        st.markdown("### Sentiment Analysis")
        
        @st.cache_data
        def analyze_sentiment(feedback_series):
            sia = SentimentIntensityAnalyzer()
            sentiments = []
            
            for feedback in feedback_series.dropna():
                if isinstance(feedback, str) and feedback.strip():
                    scores = sia.polarity_scores(feedback)
                    compound = scores['compound']
                    
                    # Classify sentiment
                    if compound >= 0.05:
                        sentiment = 'Positive'
                        sentiment_score = 1
                    elif compound <= -0.05:
                        sentiment = 'Negative'
                        sentiment_score = -1
                    else:
                        sentiment = 'Neutral'
                        sentiment_score = 0
                    
                    sentiments.append({
                        'feedback': feedback,
                        'sentiment': sentiment,
                        'compound_score': compound,
                        'sentiment_score': sentiment_score,
                        'pos': scores['pos'],
                        'neg': scores['neg'],
                        'neu': scores['neu']
                    })
            
            return pd.DataFrame(sentiments)
        
        # Perform sentiment analysis
        sentiment_df = analyze_sentiment(dff["Feedback on Fest"])
        
        if not sentiment_df.empty:
            # Sentiment distribution pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = sentiment_df['sentiment'].value_counts()
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positive': '#4CAF50',
                        'Neutral': '#FFC107', 
                        'Negative': '#FF5722'
                    }
                )
                fig_sentiment.update_layout(
                    paper_bgcolor="white", 
                    font_color="#2e7d32",
                    height=350
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                # Average sentiment by rating
                rating_sentiment = dff.merge(sentiment_df[['feedback', 'sentiment']], 
                                           left_on='Feedback on Fest', right_on='feedback', how='left')
                avg_sentiment_by_rating = rating_sentiment.groupby('Rating')['sentiment'].value_counts().unstack().fillna(0)
                
                fig_rating_sentiment = px.bar(
                    avg_sentiment_by_rating,
                    title="Sentiment Distribution by Rating",
                    color_discrete_map={
                        'Positive': '#4CAF50',
                        'Neutral': '#FFC107', 
                        'Negative': '#FF5722'
                    }
                )
                fig_rating_sentiment.update_layout(
                    paper_bgcolor="white",
                    plot_bgcolor="white", 
                    font_color="#2e7d32",
                    height=350,
                    xaxis_title="Rating",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_rating_sentiment, use_container_width=True)
            
            # Sentiment over time (if we had dates) or by event
            st.markdown("### Sentiment by Event")
            event_sentiment = dff.merge(sentiment_df[['feedback', 'sentiment']], 
                                      left_on='Feedback on Fest', right_on='feedback', how='left')
            event_sentiment_counts = event_sentiment.groupby(['Event Name', 'sentiment']).size().unstack().fillna(0)
            
            fig_event_sentiment = px.bar(
                event_sentiment_counts,
                title="Sentiment Distribution by Event",
                barmode='stack',
                color_discrete_map={
                    'Positive': '#4CAF50',
                    'Neutral': '#FFC107', 
                    'Negative': '#FF5722'
                }
            )
            fig_event_sentiment.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white", 
                font_color="#2e7d32",
                height=400,
                xaxis_title="Event",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_event_sentiment, use_container_width=True)
            
            # Sentiment summary metrics
            st.markdown("### Sentiment Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            total_feedbacks = len(sentiment_df)
            positive_pct = (sentiment_df['sentiment'] == 'Positive').sum() / total_feedbacks * 100
            neutral_pct = (sentiment_df['sentiment'] == 'Neutral').sum() / total_feedbacks * 100
            negative_pct = (sentiment_df['sentiment'] == 'Negative').sum() / total_feedbacks * 100
            
            with col1:
                st.metric("Total Feedback", f"{total_feedbacks}")
            with col2:
                st.metric("Positive", f"{positive_pct:.1f}%", 
                         delta=f"{positive_pct:.1f}%" if positive_pct > 50 else None)
            with col3:
                st.metric("Neutral", f"{neutral_pct:.1f}%")
            with col4:
                st.metric("Negative", f"{negative_pct:.1f}%",
                         delta=f"-{negative_pct:.1f}%" if negative_pct > 10 else None)
        else:
            st.info("No feedback text available for sentiment analysis.")

    # Feedback sentiment-category count
    st.markdown('<div class="section-title">Feedback Theme Distribution</div>', unsafe_allow_html=True)
    fb_cnt = dff["Feedback on Fest"].value_counts().reset_index()
    fb_cnt.columns = ["Feedback", "Count"]
    fig_fb = px.pie(fb_cnt, names="Feedback", values="Count",
                    title="Feedback Theme Share",
                    color_discrete_sequence=px.colors.qualitative.Set3)
    fig_fb.update_traces(textposition="inside", textinfo="percent+label")
    fig_fb.update_layout(paper_bgcolor="white", font_color="#2e7d32",
                         height=500, showlegend=True,
                         legend=dict(font=dict(size=10)))
    st.plotly_chart(fig_fb, use_container_width=True)

    # Avg rating per feedback theme (top 10)
    avg_fb = dff.groupby("Feedback on Fest")["Rating"].mean().reset_index()
    avg_fb.columns = ["Feedback", "Avg Rating"]
    avg_fb = avg_fb.sort_values("Avg Rating", ascending=False).head(12)
    fig_fbr = px.bar(avg_fb, x="Avg Rating", y="Feedback", orientation="h",
                     color="Avg Rating", color_continuous_scale="RdYlGn",
                     range_color=[1,5], title="Avg Rating per Feedback Theme")
    fig_fbr.update_layout(showlegend=False, plot_bgcolor="white",
                           paper_bgcolor="white", font_color="#2e7d32",
                           coloraxis_showscale=False, height=420)
    st.plotly_chart(fig_fbr, use_container_width=True)

    # Raw feedback table with filters
    st.markdown('<div class="section-title">📋 Participant Feedback Table</div>', unsafe_allow_html=True)
    
    # Interactive filters for the table
    col1, col2 = st.columns([1, 1])
    with col1:
        min_r, max_r = st.slider("Filter by Rating", 1, 5, (1, 5), help="Filter feedback by rating range")
    with col2:
        search_term = st.text_input("🔍 Search Feedback", "", help="Search in feedback text")
    
    # Apply filters
    filtered_table = dff[dff["Rating"].between(min_r, max_r)]
    if search_term:
        filtered_table = filtered_table[
            filtered_table["Feedback on Fest"].str.lower().str.contains(search_term.lower(), na=False)
        ]
    
    table_df = filtered_table[
        ["Student Name","College","State","Event Name","Feedback on Fest","Rating"]
    ].reset_index(drop=True)
    
    st.dataframe(table_df, use_container_width=True, height=340)
    
    # Summary stats for filtered feedback
    if len(filtered_table) > 0:
        st.markdown(f"**Showing {len(filtered_table)} feedback entries** (filtered from {len(dff)} total)")
        avg_filtered_rating = filtered_table['Rating'].mean()
        st.metric("Average Rating (Filtered)", f"{avg_filtered_rating:.2f} ⭐")

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — Data Explorer
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🔍 Data Explorer & Insights</div>', unsafe_allow_html=True)
    
    # Data Overview
    with st.expander("📊 Dataset Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(dff))
        with col2:
            st.metric("Columns", len(dff.columns))
        with col3:
            st.metric("Data Types", len(dff.dtypes.unique()))
        with col4:
            memory_usage = dff.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    
    # Column Analysis
    with st.expander("📋 Column Analysis", expanded=False):
        selected_column = st.selectbox("Select column to analyze:", dff.columns.tolist())
        
        if selected_column:
            col_data = dff[selected_column]
            st.markdown(f"### Analysis of `{selected_column}`")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Data Type:**")
                st.code(str(col_data.dtype))
                st.markdown("**Non-null Count:**")
                st.code(f"{col_data.notna().sum()} / {len(col_data)}")
            
            with col2:
                if col_data.dtype in ['int64', 'float64']:
                    st.markdown("**Statistics:**")
                    st.code(f"Mean: {col_data.mean():.2f}\nMin: {col_data.min()}\nMax: {col_data.max()}")
                else:
                    st.markdown("**Unique Values:**")
                    st.code(f"{col_data.nunique()} unique values")
                    if col_data.nunique() <= 10:
                        st.markdown("**Value Counts:**")
                        value_counts = col_data.value_counts().head(5)
                        for val, count in value_counts.items():
                            st.code(f"{val}: {count}")
    
    # Advanced Filtering
    with st.expander("🎯 Advanced Filters", expanded=False):
        st.markdown("**Create custom data views:**")
        
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            numeric_cols = dff.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                num_filter_col = st.selectbox("Numeric Filter Column:", ["None"] + numeric_cols)
                if num_filter_col != "None":
                    min_val, max_val = st.slider(f"Range for {num_filter_col}:", 
                                               float(dff[num_filter_col].min()), 
                                               float(dff[num_filter_col].max()), 
                                               (float(dff[num_filter_col].min()), float(dff[num_filter_col].max())))
        
        with filter_col2:
            text_cols = dff.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                text_filter_col = st.selectbox("Text Search Column:", ["None"] + text_cols)
                if text_filter_col != "None":
                    search_text = st.text_input(f"Search in {text_filter_col}:")
    
    # Export Options
    with st.expander("💾 Export Options", expanded=False):
        st.markdown("**Download your filtered dataset:**")
        
        export_format = st.radio("Export format:", ["CSV", "JSON", "Excel"], horizontal=True)
        
        if st.button("📥 Generate Download Link"):
            if export_format == "CSV":
                data = dff.to_csv(index=False)
                mime = "text/csv"
                filename = "gateways_data.csv"
            elif export_format == "JSON":
                data = dff.to_json(orient="records", indent=2)
                mime = "application/json"
                filename = "gateways_data.json"
            else:  # Excel
                # Note: This would require openpyxl, but we'll use CSV for now
                data = dff.to_csv(index=False)
                mime = "text/csv"
                filename = "gateways_data.csv"
            
            st.download_button(
                label=f"⬇️ Download {export_format}",
                data=data,
                file_name=filename,
                mime=mime
            )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")

# Enhanced Footer with quick actions
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown(
        "<div style='text-align: center; color:#4a5568; font-size:0.9rem; margin-bottom: 1rem;'>"
        "🎓 GATEWAYS-2025 Analytics Dashboard · Built with ❤️ using Streamlit & Plotly"
        "<br><small>National-Level Technical Fest · Real-time Insights & Data Visualization</small>"
        "</div>", unsafe_allow_html=True
    )

with footer_col2:
    if st.button("🔄 Refresh Data", help="Reload data from source"):
        st.cache_data.clear()
        st.rerun()

with footer_col3:
    st.markdown(
        "<div style='text-align: center; margin-top: 0.5rem;'>"
        f"<small style='color:#666;'>v1.2.0</small>"
        "</div>", unsafe_allow_html=True
    )
