# GATEWAYS-2025 Analytics Dashboard

A comprehensive data visualization and analytics dashboard for the GATEWAYS-2025 National-Level Technical Fest. This application provides real-time insights into participant data, including participation trends, geographic distribution, feedback analysis, and revenue metrics.

## Features

### Core Analytics
- **Real-time KPI Metrics**: Track total participants, colleges, states covered, events, and average ratings
- **Interactive Filtering**: Filter data by event, state, and event type with instant visual updates
- **Participation Trends**: Analyze event-wise and college-wise participation patterns
- **Geographic Insights**: Interactive India map showing state-wise participation distribution
- **Revenue Analysis**: Track revenue generated per event and overall financial metrics

### Feedback & Ratings
- **Rating Distribution**: Visual breakdown of participant ratings (1-5 scale)
- **NLTK-Powered Text Analysis**: Advanced natural language processing for feedback analysis
- **Sentiment Analysis**: VADER-based sentiment classification (Positive, Neutral, Negative)
- **Sentiment Distribution**: Pie charts and bar charts showing sentiment breakdown
- **Sentiment by Rating**: Correlation between numerical ratings and sentiment analysis
- **Sentiment by Event**: Event-wise sentiment distribution analysis
- **Lemmatization**: Word normalization for better keyword extraction
- **Enhanced Stopword Removal**: Comprehensive filtering using NLTK stopwords plus custom domain-specific terms
- **Word Cloud Visualization**: Optional visual representation of feedback keywords
- **Feedback Analysis**: Keyword frequency analysis and theme distribution
- **Sentiment Insights**: Average ratings correlated with feedback themes
- **Interactive Feedback Table**: Searchable and filterable participant feedback data

### Data Exploration
- **Advanced Filtering**: Custom data views with numeric and text-based filters
- **Column Analysis**: Detailed statistics and data type information for each column
- **Export Functionality**: Download filtered datasets in CSV format
- **Dataset Overview**: Comprehensive statistics about the data structure

### User Interface
- **Dark Theme**: Modern, professional dark interface optimized for data visualization
- **Responsive Design**: Adaptive layout that works on different screen sizes
- **Interactive Charts**: Hover effects, tooltips, and dynamic updates using Plotly
- **Tabbed Navigation**: Organized content across four main sections
- **Expandable Sections**: Collapsible content areas for better space management

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

The application will automatically download required NLTK data (stopwords, tokenizers, WordNet) on first run.

### Data Requirements
The application requires a CSV file named `C5-FestDataset-fest_dataset.csv` with the following columns:
- Student Name
- College
- State
- Event Name
- Event Type
- Amount Paid
- Rating
- Feedback on Fest

## Usage

### Running the Application
1. Ensure the data file is in the same directory as the application
2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. Open your web browser to the URL displayed in the terminal (typically http://localhost:8501)

### Navigation
- **Participation Trends**: View event and college participation analytics
- **India State Map**: Explore geographic distribution of participants
- **Feedback & Ratings**: Analyze participant feedback and ratings
- **Data Explorer**: Advanced data analysis and export tools

### Filtering Data
Use the sidebar controls to filter the entire dashboard:
- Select specific events, states, or event types
- View real-time updates across all visualizations
- Export filtered data for further analysis

## Technologies Used

- **Streamlit**: Web application framework for data science
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization
- **NLTK**: Natural language processing toolkit
  - **VADER**: Sentiment analysis for social media text
  - **WordNet**: Lemmatization and morphological analysis
  - **Stopwords**: Comprehensive stopword filtering
- **WordCloud**: Word cloud visualization
- **Matplotlib**: Plotting library for additional visualizations
- **Requests**: HTTP library for fetching GeoJSON data
- **Python**: Core programming language

## Data Visualization

The dashboard includes multiple chart types:
- Bar charts for participation and rating analysis
- Pie charts for event type and feedback theme distribution
- Choropleth maps for geographic data
- Heatmaps for college-event participation matrices
- Sentiment analysis visualizations with color-coded results
- Interactive tables with search and filter capabilities

## Sentiment Analysis

The application uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis, which is specifically designed for social media and short text like feedback comments. The sentiment classification works as follows:

- **Positive**: Compound score ≥ 0.05
- **Neutral**: Compound score between -0.05 and 0.05
- **Negative**: Compound score ≤ -0.05

Sentiment analysis provides deeper insights into participant feedback beyond numerical ratings, helping identify emotional patterns and satisfaction levels across different events and demographics.

## Performance Features

- **Caching**: Data loading is cached for improved performance
- **Lazy Loading**: Charts render only when needed
- **Optimized Queries**: Efficient data processing for large datasets
- **Memory Management**: Smart data filtering to reduce memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is developed for educational and analytical purposes. Please ensure compliance with data privacy regulations when using participant data.

## Support

For technical issues or feature requests, please check the application logs and ensure all dependencies are properly installed. The application includes comprehensive error handling and user guidance throughout the interface.