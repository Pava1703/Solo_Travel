"""
STEP 2: Data Cleaning and Preprocessing
Prepares real travel reviews for sentiment analysis
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep important words for travel context
keep_words = {'not', 'no', 'nor', 'very', 'safe', 'unsafe', 'expensive', 'cheap', 'good', 'bad'}
stop_words = stop_words - keep_words

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize text"""
    if not text:
        return ""
    
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    except:
        return text

# Load data
print("="*70)
print("STEP 2: DATA PREPROCESSING")
print("="*70)
print("\nLoading data...")

df = pd.read_csv('data/travel_reviews.csv')
print(f"Original dataset: {len(df):,} reviews")

# Remove duplicates based on review text
df = df.drop_duplicates(subset=['review_text'])
print(f"After removing duplicates: {len(df):,} reviews")

# Remove reviews with missing text
df = df.dropna(subset=['review_text'])
print(f"After removing missing text: {len(df):,} reviews")

# Clean text
print("\nCleaning text...")
df['cleaned_text'] = df['review_text'].apply(clean_text)

# Tokenize and lemmatize
print("Tokenizing and lemmatizing...")
df['processed_text'] = df['cleaned_text'].apply(tokenize_and_lemmatize)

# Extract text length features
df['review_length'] = df['review_text'].astype(str).apply(len)
df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()) if x else 0)

# Filter out very short reviews (less than 5 words)
df = df[df['word_count'] >= 5]
print(f"After filtering short reviews: {len(df):,} reviews")

# Sort by date (most recent first)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date', ascending=False)

# Save preprocessed data
df.to_csv('data/preprocessed_reviews.csv', index=False)

print("\n" + "="*70)
print("✅ PREPROCESSING COMPLETE!")
print("="*70)
print(f"\nFinal dataset: {len(df):,} reviews")
if 'date' in df.columns:
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Average review length: {df['word_count'].mean():.1f} words")
print(f"Cities: {df['city'].nunique()}")

print(f"\nTop 5 cities by review count:")
print(df['city'].value_counts().head())

print(f"\n✅ Data saved to: data/preprocessed_reviews.csv")
print("\n✅ Next step: python sentiment_analysis.py")