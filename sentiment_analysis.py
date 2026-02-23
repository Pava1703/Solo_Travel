"""
STEP 3: Aspect-Based Sentiment Analysis using BERT
Analyzes sentiment for Safety, Cost, Comfort, Social aspects
"""

import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Aspect keywords for classification
ASPECT_KEYWORDS = {
    'safety': ['safe', 'safety', 'secure', 'dangerous', 'crime', 'unsafe', 'police', 
               'theft', 'risk', 'alone', 'night', 'walk', 'scared', 'fear'],
    'cost': ['expensive', 'cheap', 'price', 'cost', 'budget', 'affordable', 'money', 
             'value', 'worth', 'pay', 'dollar', 'euro', 'pound', 'overpriced'],
    'social': ['friendly', 'social', 'meet', 'people', 'hostel', 'nightlife', 'bar',
               'club', 'party', 'crowd', 'locals', 'welcoming', 'community'],
    'comfort': ['comfortable', 'convenient', 'transport', 'accommodation', 'wifi', 
                'amenities', 'facilities', 'clean', 'modern', 'easy', 'navigate']
}

def detect_aspect(text):
    """Detect which aspect a review is discussing"""
    if not isinstance(text, str):
        return 'general'
    
    text_lower = text.lower()
    aspect_scores = {}
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        aspect_scores[aspect] = score
    
    # Return aspect with highest score
    if max(aspect_scores.values()) > 0:
        return max(aspect_scores, key=aspect_scores.get)
    return 'general'

def analyze_sentiment_batch(texts, batch_size=8):
    """Analyze sentiment for multiple texts efficiently"""
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
        batch = texts[i:i+batch_size]
        # Truncate long texts to avoid errors
        batch = [text[:512] if isinstance(text, str) else "" for text in batch]
        
        try:
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)
        except Exception as e:
            # If batch fails, add neutral sentiment
            print(f"\nError in batch {i}: {e}")
            results.extend([{'label': 'POSITIVE', 'score': 0.5}] * len(batch))
    
    return results

def sentiment_to_score(label, confidence):
    """Convert BERT sentiment to 1-5 rating scale"""
    if label == 'POSITIVE':
        # Map confidence 0.5-1.0 to rating 3.0-5.0
        return 3.0 + (confidence - 0.5) * 4.0
    else:  # NEGATIVE
        # Map confidence 0.5-1.0 to rating 1.0-3.0
        return 3.0 - (confidence - 0.5) * 4.0

# Load BERT model
print("="*70)
print("STEP 3: SENTIMENT ANALYSIS WITH BERT")
print("="*70)
print("\nLoading BERT model...")
print("(First run may take a few minutes to download the model)")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

print("✅ Model loaded successfully\n")

# Load preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv('data/preprocessed_reviews.csv')
print(f"Total reviews: {len(df):,}\n")

# Detect aspects
print("Detecting aspects in reviews...")
df['detected_aspect'] = df['cleaned_text'].apply(detect_aspect)

print("\nAspect distribution:")
print(df['detected_aspect'].value_counts())

# Analyze sentiment
print(f"\nAnalyzing sentiment for {len(df):,} reviews...")
print("This will take 2-5 minutes...\n")

sentiment_results = analyze_sentiment_batch(df['cleaned_text'].tolist(), batch_size=8)

# Extract results
df['bert_sentiment_label'] = [r['label'] for r in sentiment_results]
df['bert_confidence'] = [r['score'] for r in sentiment_results]
df['bert_sentiment_score'] = df.apply(
    lambda row: sentiment_to_score(row['bert_sentiment_label'], row['bert_confidence']),
    axis=1
)

# Calculate aspect-level scores by city
print("\nCalculating aspect-level scores by city...")

aspect_scores = df.groupby(['city', 'detected_aspect']).agg({
    'bert_sentiment_score': 'mean',
    'review_id': 'count'
}).reset_index()

aspect_scores.columns = ['city', 'aspect', 'sentiment_score', 'review_count']

# Pivot to wide format (city | safety | cost | comfort | social)
aspect_scores_wide = aspect_scores.pivot(
    index='city',
    columns='aspect',
    values='sentiment_score'
).reset_index()

# Fill missing values with overall average (only numeric columns)
numeric_cols = aspect_scores_wide.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    aspect_scores_wide[col] = aspect_scores_wide[col].fillna(aspect_scores_wide[col].mean())

# Round to 1 decimal place
aspect_scores_wide[numeric_cols] = aspect_scores_wide[numeric_cols].round(1)

# Save results
df.to_csv('data/sentiment_analyzed_reviews.csv', index=False)
aspect_scores.to_csv('data/aspect_sentiment_scores.csv', index=False)
aspect_scores_wide.to_csv('data/city_aspect_scores.csv', index=False)

print("\n" + "="*70)
print("✅ SENTIMENT ANALYSIS COMPLETE!")
print("="*70)
print(f"\nAnalyzed {len(df):,} reviews across {df['city'].nunique()} cities")

print(f"\nSentiment distribution:")
print(df['bert_sentiment_label'].value_counts())

print(f"\n\nCity Aspect Scores (Sample):")
print(aspect_scores_wide.head(10).to_string(index=False))

print(f"\n✅ Saved files:")
print(f"  - data/sentiment_analyzed_reviews.csv")
print(f"  - data/aspect_sentiment_scores.csv")
print(f"  - data/city_aspect_scores.csv")

print("\n✅ Next step: python behavior_modeling.py")