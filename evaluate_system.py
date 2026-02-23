"""
Step 9: System Evaluation
Evaluate recommendation quality using precision, recall, and user satisfaction metrics
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from recommendation_engine import TravelRecommendationEngine

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
print("Loading data and models...")
df = pd.read_csv('data/sentiment_analyzed_reviews.csv')
engine = TravelRecommendationEngine()

print("\n" + "="*70)
print("SYSTEM EVALUATION REPORT")
print("="*70)

# ============================================================================
# 1. SENTIMENT ANALYSIS EVALUATION
# ============================================================================
print("\n\n1. SENTIMENT ANALYSIS PERFORMANCE")
print("-" * 70)

# For evaluation, we use the ground truth sentiment from our data generation
y_true = (df['sentiment'] == 'positive').astype(int)
y_pred = (df['bert_sentiment_label'] == 'POSITIVE').astype(int)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}  (Target: > 0.80)")
print(f"Recall:    {recall:.3f}  (Target: > 0.80)")
print(f"F1-Score:  {f1:.3f}  (Target: > 0.80)")

if f1 > 0.80:
    print("✓ PASSED: F1-Score exceeds target threshold")
else:
    print("⚠ WARNING: F1-Score below target threshold")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Sentiment Analysis Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved confusion_matrix.png")

# ============================================================================
# 2. ASPECT EXTRACTION EVALUATION
# ============================================================================
print("\n\n2. ASPECT EXTRACTION PERFORMANCE")
print("-" * 70)

aspect_accuracy = (df['detected_aspect'] == df['aspect']).mean()
print(f"Aspect Detection Accuracy: {aspect_accuracy:.3f}")

aspect_distribution = df['detected_aspect'].value_counts()
print("\nAspect Distribution:")
print(aspect_distribution)

# ============================================================================
# 3. BEHAVIOR MODELING EVALUATION
# ============================================================================
print("\n\n3. BEHAVIOR MODELING (CLUSTERING) PERFORMANCE")
print("-" * 70)

user_profiles = pd.read_csv('data/user_profiles.csv')

# Silhouette score (already calculated in behavior_modeling.py)
from sklearn.metrics import silhouette_score
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

features = user_profiles[['safety_importance', 'cost_importance', 'social_importance', 'comfort_importance']]
features_scaled = scaler.transform(features)

silhouette = silhouette_score(features_scaled, user_profiles['cluster'])
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Number of Clusters: {user_profiles['cluster'].nunique()}")

cluster_sizes = user_profiles['cluster_name'].value_counts()
print("\nCluster Distribution:")
print(cluster_sizes)

# ============================================================================
# 4. RECOMMENDATION QUALITY EVALUATION
# ============================================================================
print("\n\n4. RECOMMENDATION QUALITY METRICS")
print("-" * 70)

# Test multiple user profiles
test_profiles = [
    {'name': 'Safety-Conscious', 'prefs': {'safety': 5.0, 'cost': 3.0, 'social': 2.0, 'comfort': 4.0}},
    {'name': 'Budget Traveler', 'prefs': {'safety': 3.0, 'cost': 5.0, 'social': 4.0, 'comfort': 2.0}},
    {'name': 'Social Explorer', 'prefs': {'safety': 3.0, 'cost': 3.0, 'social': 5.0, 'comfort': 3.0}},
    {'name': 'Comfort Seeker', 'prefs': {'safety': 4.0, 'cost': 2.0, 'social': 2.0, 'comfort': 5.0}},
    {'name': 'Balanced', 'prefs': {'safety': 4.0, 'cost': 4.0, 'social': 4.0, 'comfort': 4.0}}
]

recommendation_scores = []

for profile in test_profiles:
    recs, cluster = engine.recommend_destinations(profile['prefs'], top_n=10)
    avg_score = np.mean([r['overall_score'] for r in recs])
    recommendation_scores.append(avg_score)
    print(f"\n{profile['name']:20s} → Avg Score: {avg_score:.3f} | Type: {cluster}")

overall_rec_quality = np.mean(recommendation_scores)
print(f"\n{'Overall Recommendation Quality:'} {overall_rec_quality:.3f}/5.0")

# ============================================================================
# 5. SYSTEM PERFORMANCE METRICS
# ============================================================================
print("\n\n5. SYSTEM PERFORMANCE SUMMARY")
print("-" * 70)

metrics_summary = {
    'Sentiment Analysis F1-Score': f1,
    'Aspect Detection Accuracy': aspect_accuracy,
    'Clustering Silhouette Score': silhouette,
    'Avg Recommendation Quality': overall_rec_quality / 5.0  # Normalize to 0-1
}

print("\nMetric                          | Score  | Target | Status")
print("-" * 70)
print(f"Sentiment F1-Score              | {metrics_summary['Sentiment Analysis F1-Score']:.3f}  | > 0.80 | {'✓ PASS' if metrics_summary['Sentiment Analysis F1-Score'] > 0.80 else '✗ FAIL'}")
print(f"Aspect Detection Accuracy       | {metrics_summary['Aspect Detection Accuracy']:.3f}  | > 0.70 | {'✓ PASS' if metrics_summary['Aspect Detection Accuracy'] > 0.70 else '✗ FAIL'}")
print(f"Clustering Quality              | {metrics_summary['Clustering Silhouette Score']:.3f}  | > 0.30 | {'✓ PASS' if metrics_summary['Clustering Silhouette Score'] > 0.30 else '✗ FAIL'}")
print(f"Recommendation Quality          | {metrics_summary['Avg Recommendation Quality']:.3f}  | > 0.70 | {'✓ PASS' if metrics_summary['Avg Recommendation Quality'] > 0.70 else '✗ FAIL'}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n\n6. GENERATING EVALUATION VISUALIZATIONS")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Sentiment Distribution
ax1 = axes[0, 0]
sentiment_counts = df['bert_sentiment_label'].value_counts()
colors = ['#4CAF50', '#F44336']
ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Aspect Distribution
ax2 = axes[0, 1]
aspect_counts = df['detected_aspect'].value_counts()
ax2.bar(aspect_counts.index, aspect_counts.values, color='#2196F3', alpha=0.7)
ax2.set_title('Aspect Distribution', fontsize=14, fontweight='bold')
ax2.set_ylabel('Count')
ax2.set_xticklabels(aspect_counts.index, rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Cluster Distribution
ax3 = axes[1, 0]
cluster_counts = user_profiles['cluster_name'].value_counts()
ax3.barh(cluster_counts.index, cluster_counts.values, color='#9C27B0', alpha=0.7)
ax3.set_title('Traveler Archetype Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Count')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Recommendation Quality by Profile
ax4 = axes[1, 1]
profile_names = [p['name'] for p in test_profiles]
ax4.bar(profile_names, recommendation_scores, color='#FF9800', alpha=0.7)
ax4.set_title('Recommendation Quality by User Type', fontsize=14, fontweight='bold')
ax4.set_ylabel('Average Score (out of 5)')
ax4.set_xticklabels(profile_names, rotation=45, ha='right')
ax4.axhline(y=3.5, color='r', linestyle='--', alpha=0.5, label='Baseline')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('models/evaluation_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved evaluation_summary.png")

# ============================================================================
# 7. SAVE EVALUATION RESULTS
# ============================================================================
results_df = pd.DataFrame({
    'Metric': list(metrics_summary.keys()),
    'Score': list(metrics_summary.values())
})
results_df.to_csv('models/evaluation_results.csv', index=False)
print("✓ Saved evaluation_results.csv")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print("\n📊 Summary:")
print(f"  - Total Reviews Analyzed: {len(df):,}")
print(f"  - Cities Evaluated: {df['city'].nunique()}")
print(f"  - Traveler Archetypes: {user_profiles['cluster'].nunique()}")
print(f"  - Overall System Performance: {np.mean(list(metrics_summary.values())):.1%}")
print("\n✓ All evaluation metrics saved to models/")