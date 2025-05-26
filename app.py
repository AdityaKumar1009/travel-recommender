import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("Travel Destination Recommender")

class TravelRecommendationSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_names = ['budget', 'adventure', 'culture', 'relaxation', 
                            'nightlife', 'nature', 'food', 'shopping']
        self._load_data()
        self._train_model()
    
    def _load_data(self):
        # Sample user data for training
        users = [
            [5, 4, 2, 1, 5, 2, 4, 3],  # Luxury party traveler
            [2, 1, 5, 4, 1, 3, 5, 2],  # Cultural food enthusiast
            [4, 5, 3, 2, 2, 5, 3, 1],  # Adventure nature seeker
            [3, 2, 4, 5, 1, 4, 3, 2],  # Relaxation wellness traveler
            [5, 3, 1, 2, 4, 1, 2, 5],  # Luxury shopper
            [1, 1, 5, 5, 1, 3, 4, 1],  # Budget cultural traveler
            [4, 5, 2, 1, 3, 5, 2, 1],  # Adventure seeker
            [3, 2, 4, 4, 2, 4, 4, 3],  # Balanced traveler
        ]
        
        self.user_data = np.array(users)
        
        # Simple destination data
        self.destinations = {
            0: [  # Luxury Party Travelers
                {'name': 'Ibiza, Spain', 'rating': 4.7, 'highlights': 'Beach clubs, nightlife, luxury resorts'},
                {'name': 'Dubai, UAE', 'rating': 4.6, 'highlights': 'Luxury shopping, fine dining, modern city'},
                {'name': 'Miami, USA', 'rating': 4.5, 'highlights': 'Beach parties, art deco, vibrant nightlife'}
            ],
            1: [  # Cultural Food Enthusiasts
                {'name': 'Kyoto, Japan', 'rating': 4.8, 'highlights': 'Ancient temples, traditional cuisine, culture'},
                {'name': 'Rome, Italy', 'rating': 4.7, 'highlights': 'Historical sites, Italian food, museums'},
                {'name': 'Istanbul, Turkey', 'rating': 4.6, 'highlights': 'Rich history, Turkish cuisine, bazaars'}
            ],
            2: [  # Adventure Nature Seekers
                {'name': 'Queenstown, New Zealand', 'rating': 4.9, 'highlights': 'Adventure sports, mountains, nature'},
                {'name': 'Patagonia, Chile', 'rating': 4.8, 'highlights': 'Hiking, glaciers, wilderness'},
                {'name': 'Costa Rica', 'rating': 4.7, 'highlights': 'Wildlife, zip-lining, rainforests'}
            ],
            3: [  # Relaxation & Wellness
                {'name': 'Bali, Indonesia', 'rating': 4.6, 'highlights': 'Yoga retreats, temples, spa treatments'},
                {'name': 'Tuscany, Italy', 'rating': 4.7, 'highlights': 'Wine tours, countryside, relaxation'},
                {'name': 'Santorini, Greece', 'rating': 4.8, 'highlights': 'Sunset views, peaceful, romantic'}
            ]
        }
        
        self.cluster_names = {
            0: "Luxury Party Travelers",
            1: "Cultural Food Enthusiasts", 
            2: "Adventure Nature Seekers",
            3: "Relaxation & Wellness Travelers"
        }
    
    def _train_model(self):
        scaled_features = self.scaler.fit_transform(self.user_data)
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.kmeans.fit(scaled_features)
    
    def predict_cluster(self, preferences):
        user_features = np.array([preferences]).reshape(1, -1)
        scaled_features = self.scaler.transform(user_features)
        cluster = self.kmeans.predict(scaled_features)[0]
        return cluster
    
    def get_recommendations(self, preferences):
        cluster = self.predict_cluster(preferences)
        destinations = self.destinations[cluster]
        cluster_name = self.cluster_names[cluster]
        return destinations, cluster, cluster_name

# Initialize system
@st.cache_resource
def load_system():
    return TravelRecommendationSystem()

recommender = load_system()

# Quiz questions
st.header("Travel Preference Quiz")
st.write("Rate each aspect from 1 (not important) to 5 (very important)")

with st.form("travel_quiz"):
    budget = st.select_slider("Budget Level", options=[1,2,3,4,5], value=3,
                             help="1=Budget, 5=Luxury")
    
    adventure = st.select_slider("Adventure Level", options=[1,2,3,4,5], value=3,
                                help="1=Peaceful, 5=Extreme")
    
    culture = st.select_slider("Cultural Interest", options=[1,2,3,4,5], value=3,
                              help="1=Not interested, 5=Very interested")
    
    relaxation = st.select_slider("Relaxation Preference", options=[1,2,3,4,5], value=3,
                                 help="1=Always active, 5=Pure relaxation")
    
    nightlife = st.select_slider("Nightlife Importance", options=[1,2,3,4,5], value=3,
                                help="1=Early to bed, 5=Party every night")
    
    nature = st.select_slider("Nature Love", options=[1,2,3,4,5], value=3,
                             help="1=City person, 5=Nature purist")
    
    food = st.select_slider("Food Interest", options=[1,2,3,4,5], value=3,
                           help="1=Not important, 5=Culinary explorer")
    
    shopping = st.select_slider("Shopping Interest", options=[1,2,3,4,5], value=3,
                               help="1=No shopping, 5=Shopping spree")
    
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    preferences = [budget, adventure, culture, relaxation, nightlife, nature, food, shopping]
    
    # Get recommendations
    destinations, cluster, cluster_name = recommender.get_recommendations(preferences)
    
    st.header("Your Recommendations")
    st.success(f"Your travel type: **{cluster_name}**")
    
    st.subheader("Top Destinations for You:")
    
    for i, dest in enumerate(destinations, 1):
        st.write(f"**{i}. {dest['name']}**")
        st.write(f"Rating: {dest['rating']}/5")
        st.write(f"Highlights: {dest['highlights']}")
        st.write("---")
    
    # Show user preferences
    if st.checkbox("Show my preference profile"):
        st.subheader("Your Preferences:")
        pref_data = {
            'Aspect': ['Budget', 'Adventure', 'Culture', 'Relaxation', 'Nightlife', 'Nature', 'Food', 'Shopping'],
            'Score': preferences
        }
        df = pd.DataFrame(pref_data)
        st.bar_chart(df.set_index('Aspect'))

st.write("---")
st.write("*This is a prototype travel recommendation system using K-means clustering.*")