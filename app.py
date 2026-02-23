"""
Step 7: Flask Backend API
Provides RESTful endpoints for the recommendation system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from recommendation_engine import TravelRecommendationEngine

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load recommendation engine
print("Loading recommendation engine...")
engine = TravelRecommendationEngine()
print("✓ Engine loaded successfully")

@app.route('/')
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Solo Travel AI - Recommendation API',
        'version': '1.0',
        'endpoints': {
            '/api/recommend': 'POST - Get destination recommendations',
            '/api/cities': 'GET - Get list of all cities',
            '/api/city/<city_name>': 'GET - Get detailed city information',
            '/api/clusters': 'GET - Get traveler archetypes'
        }
    })

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Get personalized recommendations
    
    Expected JSON body:
    {
        "safety": 4.5,
        "cost": 3.2,
        "social": 4.0,
        "comfort": 3.8,
        "exclude": ["Paris", "London"]  // optional
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['safety', 'cost', 'social', 'comfort']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            if not 1.0 <= data[field] <= 5.0:
                return jsonify({'error': f'{field} must be between 1.0 and 5.0'}), 400
        
        user_preferences = {
            'safety': float(data['safety']),
            'cost': float(data['cost']),
            'social': float(data['social']),
            'comfort': float(data['comfort'])
        }
        
        exclude_cities = data.get('exclude', [])
        top_n = data.get('top_n', 10)
        
        # Get recommendations
        recommendations, cluster_name = engine.recommend_destinations(
            user_preferences, 
            top_n=top_n, 
            exclude_visited=exclude_cities
        )
        
        return jsonify({
            'success': True,
            'traveler_type': cluster_name,
            'recommendations': recommendations,
            'user_preferences': user_preferences
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of all available cities"""
    try:
        cities = engine.city_scores['city'].tolist()
        
        city_list = []
        for city in cities:
            city_data = engine.city_scores[engine.city_scores['city'] == city].iloc[0]
            review_count = len(engine.reviews[engine.reviews['city'] == city])
            
            city_list.append({
                'name': city,
                'safety': float(city_data.get('safety', 3.5)),
                'cost': float(city_data.get('cost', 3.5)),
                'social': float(city_data.get('social', 3.5)),
                'comfort': float(city_data.get('comfort', 3.5)),
                'review_count': int(review_count)
            })
        
        return jsonify({
            'success': True,
            'count': len(city_list),
            'cities': city_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/city/<city_name>', methods=['GET'])
def get_city_details(city_name):
    """Get detailed information about a specific city"""
    try:
        # Get city scores
        city_data = engine.city_scores[engine.city_scores['city'] == city_name]
        
        if city_data.empty:
            return jsonify({'error': 'City not found'}), 404
        
        city_data = city_data.iloc[0]
        
        # Get reviews for this city
        city_reviews = engine.reviews[engine.reviews['city'] == city_name]
        
        # Calculate statistics
        review_stats = {
            'total_reviews': len(city_reviews),
            'avg_rating': float(city_reviews['rating'].mean()),
            'positive_reviews': int(len(city_reviews[city_reviews['bert_sentiment_label'] == 'POSITIVE'])),
            'negative_reviews': int(len(city_reviews[city_reviews['bert_sentiment_label'] == 'NEGATIVE']))
        }
        
        # Get top reviews (highest confidence)
        top_reviews = city_reviews.nlargest(5, 'bert_confidence')[
            ['review_text', 'rating', 'detected_aspect', 'bert_sentiment_label']
        ].to_dict('records')
        
        return jsonify({
            'success': True,
            'city': city_name,
            'scores': {
                'safety': float(city_data.get('safety', 3.5)),
                'cost': float(city_data.get('cost', 3.5)),
                'social': float(city_data.get('social', 3.5)),
                'comfort': float(city_data.get('comfort', 3.5))
            },
            'statistics': review_stats,
            'sample_reviews': top_reviews
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Get information about traveler archetypes"""
    try:
        cluster_info = [
            {
                'id': 0,
                'name': 'Safety-First Planner',
                'description': 'Prioritizes safety and security above all else',
                'characteristics': ['Careful planning', 'Risk-averse', 'Values security']
            },
            {
                'id': 1,
                'name': 'Budget Explorer',
                'description': 'Focuses on affordable travel and value for money',
                'characteristics': ['Cost-conscious', 'Seeks deals', 'Flexible with comfort']
            },
            {
                'id': 2,
                'name': 'Social Butterfly',
                'description': 'Loves meeting people and social experiences',
                'characteristics': ['Outgoing', 'Enjoys hostels', 'Seeks nightlife']
            },
            {
                'id': 3,
                'name': 'Comfort Seeker',
                'description': 'Values convenience and comfort in travel',
                'characteristics': ['Prefers amenities', 'Values convenience', 'Quality-focused']
            },
            {
                'id': 4,
                'name': 'Balanced Traveler',
                'description': 'Seeks balance across all aspects',
                'characteristics': ['Well-rounded', 'Flexible', 'Adaptable']
            }
        ]
        
        return jsonify({
            'success': True,
            'clusters': cluster_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SOLO TRAVEL AI - BACKEND SERVER")
    print("="*60)
    print("\n✓ Server starting on http://localhost:5000")
    print("✓ API documentation: http://localhost:5000/")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)