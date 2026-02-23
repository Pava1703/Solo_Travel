"""
STEP 4: Behavior Modeling - Traveler Archetypes
Uses K-Means clustering to identify traveler types
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*70)
print("STEP 4: BEHAVIOR MODELING - TRAVELER ARCHETYPES")
print("="*70)

# Load sentiment analyzed data
print("\nLoading data...")
df = pd.read_csv('data/sentiment_analyzed_reviews.csv')

# Create user behavior profiles
print("Creating user behavior profiles...")

# Simulate user preferences based on their review patterns
np.random.seed(42)

# Create profiles by aggregating aspect preferences
user_features = []
user_ids = []

# Sample 200 unique user profiles
for i in range(200):
    # For each profile, sample some reviews from dataset
    sample_reviews = df.sample(n=min(10, len(df)), random_state=i)
    
    # Calculate aspect preferences
    safety_pref = sample_reviews[sample_reviews['detected_aspect'] == 'safety']['bert_sentiment_score'].mean()
    cost_pref = sample_reviews[sample_reviews['detected_aspect'] == 'cost']['bert_sentiment_score'].mean()
    social_pref = sample_reviews[sample_reviews['detected_aspect'] == 'social']['bert_sentiment_score'].mean()
    comfort_pref = sample_reviews[sample_reviews['detected_aspect'] == 'comfort']['bert_sentiment_score'].mean()
    
    # Fill NaN with neutral value
    safety_pref = safety_pref if not pd.isna(safety_pref) else 3.5
    cost_pref = cost_pref if not pd.isna(cost_pref) else 3.5
    social_pref = social_pref if not pd.isna(social_pref) else 3.5
    comfort_pref = comfort_pref if not pd.isna(comfort_pref) else 3.5
    
    # Add some variation
    safety_pref += np.random.normal(0, 0.3)
    cost_pref += np.random.normal(0, 0.3)
    social_pref += np.random.normal(0, 0.3)
    comfort_pref += np.random.normal(0, 0.3)
    
    user_features.append([safety_pref, cost_pref, social_pref, comfort_pref])
    user_ids.append(f'USER_{i:04d}')

# Create DataFrame
user_df = pd.DataFrame(
    user_features,
    columns=['safety_importance', 'cost_importance', 'social_importance', 'comfort_importance']
)
user_df['user_id'] = user_ids

print(f"Created {len(user_df)} user profiles")

# Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(
    user_df[['safety_importance', 'cost_importance', 'social_importance', 'comfort_importance']]
)

# Find optimal number of clusters
print("\nFinding optimal number of clusters...")
silhouette_scores = []
K_range = range(3, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels)
    silhouette_scores.append(score)
    print(f"  k={k}: silhouette score = {score:.3f}")

# Use 5 clusters (based on literature)
optimal_k = 5
print(f"\nUsing {optimal_k} clusters...")

# Perform clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
user_df['cluster'] = kmeans.fit_predict(features_scaled)

# Assign meaningful names based on cluster characteristics
cluster_stats = user_df.groupby('cluster')[
    ['safety_importance', 'cost_importance', 'social_importance', 'comfort_importance']
].mean()

# Name clusters based on dominant characteristic
cluster_names = {}
for cluster_id in range(optimal_k):
    scores = cluster_stats.loc[cluster_id]
    max_aspect = scores.idxmax().replace('_importance', '')
    
    if max_aspect == 'safety':
        cluster_names[cluster_id] = 'Safety-First Planner'
    elif max_aspect == 'cost':
        cluster_names[cluster_id] = 'Budget Explorer'
    elif max_aspect == 'social':
        cluster_names[cluster_id] = 'Social Butterfly'
    elif max_aspect == 'comfort':
        cluster_names[cluster_id] = 'Comfort Seeker'
    else:
        cluster_names[cluster_id] = 'Balanced Traveler'

# Handle duplicates
used_names = []
for cluster_id, name in cluster_names.items():
    if name in used_names:
        cluster_names[cluster_id] = 'Balanced Traveler'
    used_names.append(cluster_names[cluster_id])

user_df['cluster_name'] = user_df['cluster'].map(cluster_names)

print("\n" + "="*70)
print("TRAVELER ARCHETYPES IDENTIFIED:")
print("="*70)
for cluster_id, name in cluster_names.items():
    count = len(user_df[user_df['cluster'] == cluster_id])
    print(f"\n{name} (Cluster {cluster_id}, n={count}):")
    print(cluster_stats.loc[cluster_id].round(2).to_string())

# Visualize clusters
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

aspects = ['safety_importance', 'cost_importance', 'social_importance', 'comfort_importance']
titles = ['Safety Importance', 'Cost Importance', 'Social Importance', 'Comfort Importance']

for idx, (aspect, title) in enumerate(zip(aspects, titles)):
    ax = axes[idx // 2, idx % 2]
    
    for cluster_id in range(optimal_k):
        cluster_data = user_df[user_df['cluster'] == cluster_id]
        ax.scatter(
            cluster_data.index, 
            cluster_data[aspect],
            label=cluster_names[cluster_id],
            alpha=0.6,
            s=50
        )
    
    ax.set_xlabel('User Index')
    ax.set_ylabel(title)
    ax.set_title(f'{title} by Traveler Type')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/traveler_archetypes.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved visualization: models/traveler_archetypes.png")

# Save models and data
with open('models/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

user_df.to_csv('data/user_profiles.csv', index=False)
cluster_stats.to_csv('data/cluster_characteristics.csv')

silhouette_final = silhouette_score(features_scaled, user_df['cluster'])

print(f"\n✅ Saved files:")
print(f"  - models/kmeans_model.pkl")
print(f"  - models/scaler.pkl")
print(f"  - data/user_profiles.csv")
print(f"  - data/cluster_characteristics.csv")

print(f"\n📊 Performance Metrics:")
print(f"  - Silhouette Score: {silhouette_final:.3f}")
print(f"  - Number of Clusters: {optimal_k}")
print(f"  - Total Users Profiled: {len(user_df)}")

print("\n✅ Next step: python recommendation_engine.py")