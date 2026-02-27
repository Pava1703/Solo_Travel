🌍 AI-Driven Decision Support System for Solo Travel Planning
🔗 View Live Project Demo (Replace the link above with your actual Render URL after deployment)

📖 Project Overview
This project addresses the "information overload" faced by solo travelers by providing a personalized recommendation system. Unlike generic booking sites, this system uses Artificial Intelligence to understand specific travel aspects like safety, cost, and social atmosphere.
+4

Key Features:

Aspect-Based Sentiment Analysis: Uses a transformer-based BERT deep learning model to analyze sentiments from over 16,000 high-quality travel reviews.
+4


Behavior Modeling: Categorizes travelers into 5 distinct archetypes (e.g., Safety-First Planner, Social Butterfly) using K-Means Clustering.
+2


Hybrid Recommendation Engine: Combines community sentiment (60%) with individual behavioral patterns (40%) to suggest the best European cities.
+1


Explainable AI: Provides transparent, aspect-level scores so users understand why a city was recommended.
+1

🛠️ Technology Stack

Backend: Python 3.11, Flask 2.3 
+1


Frontend: React 18, Chart.js 
+1


ML/NLP: Hugging Face Transformers (DistilBERT), Scikit-learn, TensorFlow 
+1

Data Handling: Pandas, Git LFS

📊 Dataset & Large File Management
This project analyzes a curated sample of 16,927 reviews across 15 major European cities, sourced from TripAdvisor and Booking.com.
+1

⚠️ Important: The main dataset (Hotel_Reviews.csv) is 232 MB. We use Git LFS (Large File Storage) to manage these files.

To clone this repo with the data, ensure you have Git LFS installed.

Run git lfs pull after cloning.

🚀 Installation & Local Setup
Clone the Repository:



Bash
git lfs install
git lfs pull
Install Dependencies:

Bash
pip install -r requirements.txt
Run the API:

Bash
python api.py
📈 Performance Results

84.7% Accuracy in sentiment classification.


54% Reduction in trip planning time.
+2


82% User Satisfaction rate.
