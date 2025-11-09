import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
def add_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .section-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .positive-sentiment {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2c3e50;
    }
    
    .negative-sentiment {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #2c3e50;
    }
    
    .neutral-sentiment {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: transform 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .sentiment-result {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Database setup
def init_database():
    conn = sqlite3.connect('sentiment_history.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sentiment_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT NOT NULL,
                  sentiment TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn

# Initialize sentiment analyzer
@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

# Sentiment analysis functions
def analyze_sentiment_vader(text):
    analyzer = get_sentiment_analyzer()
    scores = analyzer.polarity_scores(text)
    
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "Positive"
        confidence = compound
    elif compound <= -0.05:
        sentiment = "Negative"
        confidence = abs(compound)
    else:
        sentiment = "Neutral"
        confidence = abs(compound)
    
    return sentiment, confidence, scores

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        sentiment = "Positive"
        confidence = polarity
    elif polarity < 0:
        sentiment = "Negative"
        confidence = abs(polarity)
    else:
        sentiment = "Neutral"
        confidence = 0
    
    return sentiment, confidence

# Save to history
def save_to_history(text, sentiment, confidence, conn):
    c = conn.cursor()
    c.execute("INSERT INTO sentiment_history (text, sentiment, confidence) VALUES (?, ?, ?)",
              (text, sentiment, confidence))
    conn.commit()

# Get history
def get_history(conn, limit=50):
    c = conn.cursor()
    c.execute("SELECT text, sentiment, confidence, timestamp FROM sentiment_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    return c.fetchall()

# Create visualizations
def create_sentiment_distribution(df):
    fig = px.pie(df, names='sentiment', values='count', 
                 title='Sentiment Distribution',
                 color_discrete_map={'Positive': '#00ff00', 'Negative': '#ff0000', 'Neutral': '#ffff00'})
    return fig

def create_confidence_chart(df):
    fig = px.bar(df, x='sentiment', y='confidence', 
                 title='Average Confidence by Sentiment',
                 color='sentiment',
                 color_discrete_map={'Positive': '#00ff00', 'Negative': '#ff0000', 'Neutral': '#ffff00'})
    return fig

def create_wordcloud(text_list, sentiment_type):
    text = ' '.join(text_list)
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          colormap='Greens' if sentiment_type == 'Positive' else 'Reds').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{sentiment_type} Words Cloud', fontsize=16, fontweight='bold')
    return fig

# Main application
def main():
    add_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Tool</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        section = st.radio("Choose a section:", 
                          ["📝 Text Analysis", "📊 Dataset Analysis", "📈 History"])
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This tool analyzes text sentiment using advanced NLP techniques. You can analyze individual texts, upload datasets, or view your analysis history.")
    
    # Initialize database
    conn = init_database()
    
    # Section 1: Text Analysis
    if section == "📝 Text Analysis":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("## 📝 Single Text Analysis")
        st.markdown("Analyze the sentiment of individual text inputs")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area("Enter your text here:", 
                                      placeholder="Type or paste your text here for sentiment analysis...",
                                      height=150)
            
            analysis_method = st.selectbox("Choose analysis method:", 
                                         ["VADER Sentiment", "TextBlob", "Both (Compare)"])
            
            analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Quick Examples")
            example_texts = [
                "I love this product! It's amazing and works perfectly.",
                "This is terrible. I hate it and want my money back.",
                "The weather is okay today, nothing special."
            ]
            
            for i, example in enumerate(example_texts):
                if st.button(f"Example {i+1}", key=f"example_{i}"):
                    st.session_state.text_input = example
        
        if analyze_button and text_input:
            with st.spinner("Analyzing sentiment..."):
                # VADER Analysis
                vader_sentiment, vader_confidence, vader_scores = analyze_sentiment_vader(text_input)
                
                # TextBlob Analysis
                textblob_sentiment, textblob_confidence = analyze_sentiment_textblob(text_input)
                
                # Save to history
                save_to_history(text_input, vader_sentiment, vader_confidence, conn)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                if analysis_method == "VADER Sentiment" or analysis_method == "Both (Compare)":
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"<div class='metric-card {'positive-sentiment' if vader_sentiment == 'Positive' else 'negative-sentiment' if vader_sentiment == 'Negative' else 'neutral-sentiment'}'>", 
                                   unsafe_allow_html=True)
                        st.markdown(f"**VADER Sentiment:** {vader_sentiment}")
                        st.markdown(f"**Confidence:** {vader_confidence:.2%}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### VADER Scores")
                        st.json(vader_scores)
                    
                    with col3:
                        # Create a gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=vader_scores['compound'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Compound Score"},
                            gauge={'axis': {'range': [-1, 1]},
                                   'bar': {'color': "darkblue"},
                                   'steps': [
                                       {'range': [-1, -0.05], 'color': "red"},
                                       {'range': [-0.05, 0.05], 'color': "yellow"},
                                       {'range': [0.05, 1], 'color': "green"}],
                                   'threshold': {'line': {'color': "black", 'width': 4},
                                                'thickness': 0.75, 'value': 0}}))
                        st.plotly_chart(fig, use_container_width=True)
                
                if analysis_method == "TextBlob" or analysis_method == "Both (Compare)":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"<div class='metric-card {'positive-sentiment' if textblob_sentiment == 'Positive' else 'negative-sentiment' if textblob_sentiment == 'Negative' else 'neutral-sentiment'}'>", 
                                   unsafe_allow_html=True)
                        st.markdown(f"**TextBlob Sentiment:** {textblob_sentiment}")
                        st.markdown(f"**Confidence:** {textblob_confidence:.2%}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        if analysis_method == "Both (Compare)":
                            st.markdown("#### Comparison")
                            comparison_data = {
                                'Method': ['VADER', 'TextBlob'],
                                'Sentiment': [vader_sentiment, textblob_sentiment],
                                'Confidence': [vader_confidence, textblob_confidence]
                            }
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 2: Dataset Analysis
    elif section == "📊 Dataset Analysis":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("## 📊 Dataset Analysis")
        st.markdown("Upload and analyze sentiment in bulk datasets")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                st.markdown("### 📋 Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Let user select text column
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    selected_column = st.selectbox("Select the text column for analysis:", text_columns)
                    
                    if st.button("🔍 Analyze Dataset"):
                        with st.spinner("Analyzing dataset..."):
                            # Analyze sentiments
                            sentiments = []
                            confidences = []
                            
                            for text in df[selected_column].astype(str):
                                sentiment, confidence, _ = analyze_sentiment_vader(text)
                                sentiments.append(sentiment)
                                confidences.append(confidence)
                            
                            df['sentiment'] = sentiments
                            df['confidence'] = confidences
                            
                            # Display results
                            st.markdown("### 📊 Analysis Results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            sentiment_counts = df['sentiment'].value_counts()
                            
                            with col1:
                                st.markdown(f"<div class='metric-card positive-sentiment'>", unsafe_allow_html=True)
                                st.markdown(f"**Positive:** {sentiment_counts.get('Positive', 0)}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"<div class='metric-card negative-sentiment'>", unsafe_allow_html=True)
                                st.markdown(f"**Negative:** {sentiment_counts.get('Negative', 0)}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"<div class='metric-card neutral-sentiment'>", unsafe_allow_html=True)
                                st.markdown(f"**Neutral:** {sentiment_counts.get('Neutral', 0)}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                                st.markdown(f"**Total:** {len(df)}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Visualizations
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                sentiment_dist = df['sentiment'].value_counts().reset_index()
                                sentiment_dist.columns = ['sentiment', 'count']
                                fig = create_sentiment_distribution(sentiment_dist)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                avg_confidence = df.groupby('sentiment')['confidence'].mean().reset_index()
                                fig = create_confidence_chart(avg_confidence)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Word clouds
                            st.markdown("### ☁️ Word Clouds")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                positive_texts = df[df['sentiment'] == 'Positive'][selected_column].tolist()
                                if positive_texts:
                                    fig = create_wordcloud(positive_texts, 'Positive')
                                    st.pyplot(fig)
                            
                            with col2:
                                negative_texts = df[df['sentiment'] == 'Negative'][selected_column].tolist()
                                if negative_texts:
                                    fig = create_wordcloud(negative_texts, 'Negative')
                                    st.pyplot(fig)
                            
                            # Download results
                            st.markdown("### 💾 Download Results")
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Analyzed Dataset",
                                data=csv,
                                file_name=f"analyzed_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                else:
                    st.error("No text columns found in the dataset!")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 3: History
    elif section == "📈 History":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("## 📈 Analysis History")
        st.markdown("View and manage your sentiment analysis history")
        
        # Get history
        history = get_history(conn, limit=100)
        
        if history:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_filter = st.selectbox("Filter by sentiment:", 
                                              ["All", "Positive", "Negative", "Neutral"])
            
            with col2:
                date_filter = st.date_input("Filter by date:", value=None)
            
            with col3:
                search_term = st.text_input("Search in text:", placeholder="Enter search term...")
            
            # Convert to DataFrame for easier filtering
            history_df = pd.DataFrame(history, 
                                    columns=['Text', 'Sentiment', 'Confidence', 'Timestamp'])
            history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
            
            # Apply filters
            if sentiment_filter != "All":
                history_df = history_df[history_df['Sentiment'] == sentiment_filter]
            
            if date_filter:
                history_df = history_df[history_df['Timestamp'].dt.date == date_filter]
            
            if search_term:
                history_df = history_df[history_df['Text'].str.contains(search_term, case=False, na=False)]
            
            # Display statistics
            st.markdown("### 📊 History Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_analyses = len(history_df)
                st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"**Total Analyses:** {total_analyses}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                positive_count = len(history_df[history_df['Sentiment'] == 'Positive'])
                st.markdown(f"<div class='metric-card positive-sentiment'>", unsafe_allow_html=True)
                st.markdown(f"**Positive:** {positive_count}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                negative_count = len(history_df[history_df['Sentiment'] == 'Negative'])
                st.markdown(f"<div class='metric-card negative-sentiment'>", unsafe_allow_html=True)
                st.markdown(f"**Negative:** {negative_count}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                neutral_count = len(history_df[history_df['Sentiment'] == 'Neutral'])
                st.markdown(f"<div class='metric-card neutral-sentiment'>", unsafe_allow_html=True)
                st.markdown(f"**Neutral:** {neutral_count}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display history table
            st.markdown("### 📝 History Details")
            
            # Pagination
            items_per_page = 10
            total_pages = max(1, (len(history_df) - 1) // items_per_page + 1)
            
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col2:
                page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            
            page_data = history_df.iloc[start_idx:end_idx]
            
            for idx, row in page_data.iterrows():
                sentiment_class = f"{row['Sentiment'].lower()}-sentiment"
                st.markdown(f"<div class='sentiment-result {sentiment_class}'>", unsafe_allow_html=True)
                st.markdown(f"**Text:** {row['Text']}")
                st.markdown(f"**Sentiment:** {row['Sentiment']} | **Confidence:** {row['Confidence']:.2%}")
                st.markdown(f"**Timestamp:** {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Clear All History"):
                    c = conn.cursor()
                    c.execute("DELETE FROM sentiment_history")
                    conn.commit()
                    st.success("History cleared successfully!")
                    st.rerun()
            
            with col2:
                if st.button("📊 Export History"):
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download History CSV",
                        data=csv,
                        file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
        else:
            st.info("No history found. Start analyzing some text to see history here!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    conn.close()

if __name__ == "__main__":
    main()