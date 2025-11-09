# 🎭 Sentiment Analysis Tool

A comprehensive sentiment analysis application built with Streamlit that provides three main sections for analyzing text sentiment: single text analysis, dataset analysis, and history tracking.

## ✨ Features

### 📝 Section 1: Single Text Analysis
- **Real-time sentiment analysis** using VADER and TextBlob algorithms
- **Multiple analysis methods** with comparison capabilities
- **Interactive visualizations** including gauge charts and confidence scores
- **Quick examples** for testing different sentiment types
- **Attractive UI** with gradient backgrounds and animations

### 📊 Section 2: Dataset Analysis
- **CSV file upload** support for bulk sentiment analysis
- **Automatic column detection** for text analysis
- **Comprehensive visualizations** including pie charts, bar charts, and word clouds
- **Sentiment distribution analysis** with confidence metrics
- **Export functionality** for analyzed datasets
- **Word clouds** for positive and negative sentiments

### 📈 Section 3: History Tracking
- **SQLite database** for persistent storage
- **Advanced filtering** by sentiment, date, and search terms
- **Pagination** for large history datasets
- **Export history** to CSV format
- **Clear history** functionality
- **Statistics dashboard** with summary metrics

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project
```bash
cd sentiment_analysis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data (for TextBlob)
```bash
python -m textblob.download_corpora
```

### Step 4: Run the Application
```bash
streamlit run sentiment_analyzer.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 📝 Single Text Analysis
1. Navigate to the "Text Analysis" section
2. Enter your text in the text area
3. Choose your preferred analysis method (VADER, TextBlob, or both)
4. Click "Analyze Sentiment"
5. View detailed results with confidence scores and visualizations

### 📊 Dataset Analysis
1. Go to the "Dataset Analysis" section
2. Upload a CSV file containing text data
3. Select the column containing text for analysis
4. Click "Analyze Dataset"
5. Explore visualizations and download results

### 📈 History Management
1. Visit the "History" section
2. Use filters to find specific analyses
3. Export history for external use
4. Clear history when needed

## 🛠️ Technical Details

### Sentiment Analysis Algorithms
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: Optimized for social media text and handles emojis well
- **TextBlob**: Simple and effective for general text analysis

### Database Schema
```sql
CREATE TABLE sentiment_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### File Structure
```
sentiment_analysis/
├── sentiment_analyzer.py      # Main application file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── sentiment_history.db       # SQLite database (created automatically)
└── [CSV files for analysis]   # User-uploaded datasets
```

## 🎨 UI Features

- **Gradient animations** and modern design
- **Responsive layout** that works on all devices
- **Interactive charts** using Plotly
- **Color-coded sentiment indicators**
- **Smooth transitions** and hover effects

## 📊 Sample Datasets

You can test the application with sample datasets containing text columns. Some examples:
- Product reviews
- Social media posts
- Customer feedback
- News articles
- Survey responses

## 🔧 Customization

### Adding New Analysis Methods
To add a new sentiment analysis algorithm:
1. Create a new function in the analysis section
2. Add it to the analysis method dropdown
3. Implement the visualization logic

### Modifying UI Themes
Edit the CSS in the `add_custom_css()` function to change:
- Color schemes
- Animations
- Layout styles
- Component appearances

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**: Run `python -m textblob.download_corpora`
2. **Port Already in Use**: Change the port with `streamlit run sentiment_analyzer.py --server.port 8502`
3. **CSV File Issues**: Ensure your CSV has proper encoding (UTF-8 recommended)

### Performance Tips
- For large datasets (>10,000 rows), consider processing in batches
- Use the history pagination to avoid memory issues
- Clear old history regularly to maintain performance

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new analysis algorithms
- Improving the UI/UX
- Adding more visualization options
- Enhancing performance
- Fixing bugs or issues

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue in the repository

---

**Happy Analyzing! 🎉**
