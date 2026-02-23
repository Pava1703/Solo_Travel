"""
Step 6: Hybrid Recommendation Engine
Combines sentiment analysis + behavior modeling for personalized recommendations
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class TravelRecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine"""
        # Load data
        self.city_scores = pd.read_csv('data/city_aspect_scores.csv')
        self.user_profiles = pd.read_csv('data/user_profiles.csv')
        self.reviews = pd.read_csv('data/sentiment_analyzed_reviews.csv')
        
        # Load models
        with open('models/kmeans_model.pkl', 'rb') as f:
            self.kmeans = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("✓ Recommendation engine initialized")
    
    def get_user_cluster(self, safety_pref, cost_pref, social_pref, comfort_pref):
        """Identify user's traveler archetype based on preferences"""
        user_features = np.array([[safety_pref, cost_pref, social_pref, comfort_pref]])
        user_features_scaled = self.scaler.transform(user_features)
        cluster = self.kmeans.predict(user_features_scaled)[0]
        
        # Determine cluster name based on HIGHEST preference
        preferences = {
            'safety': safety_pref,
            'cost': cost_pref,
            'social': social_pref,
            'comfort': comfort_pref
        }
        
        # Find the highest preference
        max_pref_key = max(preferences, key=preferences.get)
        max_pref_value = preferences[max_pref_key]
        
        # Map to traveler type based on highest preference
        if max_pref_value >= 4.0:
            if max_pref_key == 'safety':
                return cluster, 'Safety-First Planner'
            elif max_pref_key == 'cost':
                return cluster, 'Budget Explorer'
            elif max_pref_key == 'social':
                return cluster, 'Social Butterfly'
            elif max_pref_key == 'comfort':
                return cluster, 'Comfort Seeker'
        
        # Check if preferences are balanced (all within 1.0 of each other)
        pref_values = list(preferences.values())
        if max(pref_values) - min(pref_values) <= 1.0:
            return cluster, 'Balanced Traveler'
        
        # Default based on highest preference
        cluster_names = {
            'safety': 'Safety-First Planner',
            'cost': 'Budget Explorer',
            'social': 'Social Butterfly',
            'comfort': 'Comfort Seeker'
        }
        
        return cluster, cluster_names.get(max_pref_key, 'Balanced Traveler')
    
    def calculate_city_score(self, city_data, user_preferences):
        """Calculate overall score for a city based on user preferences"""
        score = 0
        weights = {
            'safety': user_preferences.get('safety', 1.0),
            'cost': user_preferences.get('cost', 1.0),
            'social': user_preferences.get('social', 1.0),
            'comfort': user_preferences.get('comfort', 1.0)
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        for aspect, weight in weights.items():
            if aspect in city_data and not pd.isna(city_data[aspect]):
                score += city_data[aspect] * weight
        
        return score
    
    def get_similar_users(self, user_cluster, top_n=10):
        """Find similar users based on cluster"""
        similar_users = self.user_profiles[self.user_profiles['cluster'] == user_cluster]
        return similar_users.head(top_n)
    
    def collaborative_filtering(self, city, user_cluster):
        """Calculate collaborative filtering score"""
        # Find users in same cluster who reviewed this city
        similar_user_reviews = self.reviews[
            (self.reviews['city'] == city) & 
            (self.reviews['reviewer_type'].notna())
        ]
        
        if len(similar_user_reviews) > 0:
            return similar_user_reviews['bert_sentiment_score'].mean()
        return 3.5  # neutral score if no data
    
    def recommend_destinations(self, user_preferences, top_n=10, exclude_visited=None):
        """
        Generate personalized destination recommendations
        
        Parameters:
        - user_preferences: dict with keys 'safety', 'cost', 'social', 'comfort' (values 1-5)
        - top_n: number of recommendations to return
        - exclude_visited: list of cities to exclude
        """
        if exclude_visited is None:
            exclude_visited = []
        
        # Identify user cluster
        cluster_id, cluster_name = self.get_user_cluster(
            user_preferences['safety'],
            user_preferences['cost'],
            user_preferences['social'],
            user_preferences['comfort']
        )
        
        recommendations = []
        
        for idx, row in self.city_scores.iterrows():
            city = row['city']
            
            if city in exclude_visited:
                continue
            
            # Content-based score (sentiment analysis)
            content_score = self.calculate_city_score(row, user_preferences)
            
            # Collaborative filtering score
            collab_score = self.collaborative_filtering(city, cluster_id)
            
            # Hybrid score (weighted combination)
            hybrid_score = 0.6 * content_score + 0.4 * collab_score
            
            # Get aspect scores for explanation
            aspect_scores = {
                'safety': row.get('safety', 3.5),
                'cost': row.get('cost', 3.5),
                'social': row.get('social', 3.5),
                'comfort': row.get('comfort', 3.5)
            }
            
            # Count reviews for credibility
            review_count = len(self.reviews[self.reviews['city'] == city])
            
            recommendations.append({
                'city': city,
                'overall_score': hybrid_score,
                'content_score': content_score,
                'collaborative_score': collab_score,
                'safety_score': aspect_scores['safety'],
                'cost_score': aspect_scores['cost'],
                'social_score': aspect_scores['social'],
                'comfort_score': aspect_scores['comfort'],
                'review_count': review_count,
                'user_cluster': cluster_name
            })
        
        # Sort by overall score
        recommendations = sorted(recommendations, key=lambda x: x['overall_score'], reverse=True)
        
        return recommendations[:top_n], cluster_name

# Test the recommendation engine
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION ENGINE")
    print("="*60)
    
    # Initialize engine
    engine = TravelRecommendationEngine()
    
    # Test Case 1: Safety-conscious traveler
    print("\n\nTest Case 1: Safety-Conscious Traveler")
    print("-" * 60)
    user_prefs_1 = {
        'safety': 5.0,    # Very important
        'cost': 3.0,      # Moderate importance
        'social': 2.0,    # Less important
        'comfort': 4.0    # Important
    }
    
    recommendations_1, cluster_1 = engine.recommend_destinations(user_prefs_1, top_n=5)
    print(f"Traveler Type: {cluster_1}")
    print("\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations_1, 1):
        print(f"\n{i}. {rec['city']}")
        print(f"   Overall Score: {rec['overall_score']:.2f}/5.0")
        print(f"   Safety: {rec['safety_score']:.1f} | Cost: {rec['cost_score']:.1f} | " \
              f"Social: {rec['social_score']:.1f} | Comfort: {rec['comfort_score']:.1f}")
        print(f"   Based on {rec['review_count']} reviews")
    
    # Test Case 2: Budget + Social traveler
    print("\n\n" + "="*60)
    print("Test Case 2: Budget + Social Traveler")
    print("-" * 60)
    user_prefs_2 = {
        'safety': 3.0,
        'cost': 5.0,      # Very important
        'social': 5.0,    # Very important
        'comfort': 2.0
    }
    
    recommendations_2, cluster_2 = engine.recommend_destinations(user_prefs_2, top_n=5)
    print(f"Traveler Type: {cluster_2}")
    print("\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations_2, 1):
        print(f"\n{i}. {rec['city']}")
        print(f"   Overall Score: {rec['overall_score']:.2f}/5.0")
        print(f"   Safety: {rec['safety_score']:.1f} | Cost: {rec['cost_score']:.1f} | " \
              f"Social: {rec['social_score']:.1f} | Comfort: {rec['comfort_score']:.1f}")
        print(f"   Based on {rec['review_count']} reviews")
    
    # Save engine
    with open('models/recommendation_engine.pkl', 'wb') as f:
        pickle.dump(engine, f)
    
    print("\n\n✓ Recommendation engine saved to: models/recommendation_engine.pkl")