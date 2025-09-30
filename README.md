# Sentiment Analysis on E-Commerce Reviews

A comprehensive sentiment analysis project that analyzes Women's Clothing E-Commerce Reviews using multiple approaches: **VADER** (lexicon-based), **RoBERTa** (transformer-based), and **Rating-based** sentiment classification.

## Features

- **Multiple Sentiment Analysis Methods**:
  - VADER (Valence Aware Dictionary and sEntiment Reasoner)
  - RoBERTa (Robustly Optimized BERT Pretraining Approach)
  - Rating-based sentiment mapping
  
- **Comprehensive Analysis**:
  - Data preprocessing and cleaning
  - Text feature extraction
  - Sentiment distribution visualization
  - Method comparison and correlation analysis
  - Automated reporting

## Folder Structure

```
├── README.md                          
├── requirements.txt                   
├── outputs/                           
│   ├── df_new_first_50.csv           
│   ├── polarity.png                  
│   ├── Rating Sentiment.png          
│   ├── sample_of_data.csv           
│   └── vader_roBERTa.png             
└── sentiment_analysis/       
    ├── sentiment-analysis-vader-roberta.ipynb
    └── SRC/                          
        ├── data_preprocessing.py     
        ├── Vader.py                 
        ├── roBERTa.py                
        └── Pipeline.py               
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/akshat7776/Sentiment_Analysis_E-Commerce_Reviews.git Sentiment_Analysis_E-Commerce_Reviews
   cd Sentiment_Analysis_E-Commerce_Reviews
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not already present):
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('stopwords')
   ```

## Dataset

The project uses the **Women's Clothing E-Commerce Reviews** dataset containing:
- Customer reviews text
- Ratings (1-5 stars)
- Product information
- Customer demographics

**Key Statistics**:
- Reviews with text content after cleaning
- Rating distribution across 1-5 stars
- Text length and feature analysis

## Quick Start

### Option 1: Using the Complete Pipeline

```python
from sentiment_analysis.SRC.Pipeline import run_complete_analysis, plot_sentiment_comparison, generate_report

# Run complete analysis on your dataset
df = run_complete_analysis("Womens Clothing E-Commerce Reviews.csv")

# Generate visualizations
plot_sentiment_comparison(df)

# Print detailed report
generate_report(df)
```

### Option 2: Using Individual Modules

```python
from sentiment_analysis.SRC.data_preprocessing import prepare_dataset
from sentiment_analysis.SRC.Vader import run_vader_analysis
from sentiment_analysis.SRC.roBERTa import run_roberta_analysis

# Load and prepare data
df = prepare_dataset('Womens Clothing E-Commerce Reviews.csv')

# Run VADER analysis
df_vader = run_vader_analysis(df)

# Run RoBERTa analysis
df_complete = run_roberta_analysis(df_vader)
```

### Option 3: Using Jupyter Notebooks

Open the provided notebook:
- `sentiment_analysis/sentiment-analysis-vader-roberta.ipynb` - Main analysis notebook with complete workflow

## Analysis Methods

### 1. VADER Sentiment Analysis
- **Type**: Lexicon and rule-based
- **Strengths**: Fast, works well with social media text, handles intensifiers
- **Output**: Compound score (-1 to +1), individual neg/neu/pos scores

### 2. RoBERTa Sentiment Analysis
- **Type**: Transformer-based deep learning
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Strengths**: Context-aware, handles complex language patterns
- **Output**: Probability scores for negative/neutral/positive classes

### 3. Rating-based Sentiment
- **Type**: Rule-based mapping
- **Logic**: 
  - Ratings 1-2 → Negative
  - Rating 3 → Neutral  
  - Ratings 4-5 → Positive

## Key Visualizations

The project generates various visualizations:

1. **Sentiment Distribution Pie Charts**: Compare sentiment distributions across methods
2. **Polarity Score Distributions**: Density plots showing score distributions
3. **Rating vs Sentiment Analysis**: Box plots and bar charts
4. **Method Correlation Analysis**: Scatter plots and correlation matrices
5. **Agreement Analysis**: Confusion matrices between different methods

## Module Documentation

### `data_preprocessing.py`
- Data loading and validation
- Text cleaning and preprocessing  
- Rating-based sentiment mapping
- Complete dataset preparation pipeline

### `Vader.py`
- VADER sentiment analysis implementation
- Polarity score calculation and categorization
- Batch processing for dataframes
- Sentiment distribution analysis

### `roBERTa.py`
- RoBERTa model loading and inference
- Transformer-based sentiment classification
- Polarity calculation using weighted scores
- GPU/CPU optimization

### `Pipeline.py`
- Complete analysis workflow orchestration
- Comparative visualization generation
- Comprehensive reporting and statistics
- Single entry point for all analyses

## Configuration

The pipeline is simple and straightforward - no complex configuration needed:

```python
# Basic usage - just provide your CSV file path
df = run_complete_analysis("your_data.csv")

# The pipeline automatically:
# 1. Loads and cleans your data
# 2. Runs VADER sentiment analysis  
# 3. Runs RoBERTa sentiment analysis
# 4. Generates comparison visualizations
# 5. Creates detailed reports
```

## Results and Insights

The analysis provides insights into:
- **Sentiment Distribution**: How sentiments vary across the dataset
- **Method Agreement**: How well different methods agree on sentiment classification  
- **Rating Correlation**: How sentiment scores correlate with customer ratings
- **Model Performance**: Comparative analysis of VADER vs RoBERTa performance