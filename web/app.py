"""
Streamlit web interface for the TunedIn recommendation system.
"""
import os
import sys
import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# API URL
API_URL = f"http://{config.API_HOST}:{config.API_PORT}"

# Page configuration
st.set_page_config(
    page_title="TunedIn - AI Music Recommendations",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DB954;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #777;
        margin-top: 0;
    }
    .recommendation-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #1DB954;
    }
    .song-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .artist-name {
        font-size: 1rem;
        color: #555;
        margin-bottom: 5px;
    }
    .album-name {
        font-size: 0.9rem;
        color: #777;
        margin-bottom: 5px;
    }
    .genre-tag {
        background-color: #1DB954;
        color: white;
        padding: 3px 8px;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 5px;
    }
    .feature-value {
        font-size: 0.9rem;
        color: #555;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🎵 TunedIn</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your AI-powered guide to the perfect soundtrack</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<p class='sidebar-header'>Model Selection</p>", unsafe_allow_html=True)
model_name = st.sidebar.selectbox(
    "Select Recommendation Model",
    ["graphsage", "gcn", "gat", "lightgcn"],
    index=0
)

# Get API status
@st.cache_data(ttl=60)
def get_api_status():
    try:
        response = requests.get(f"{API_URL}/")
        return response.json()
    except:
        return {"status": "offline", "models_loaded": [], "num_users": 0, "num_songs": 0}

status = get_api_status()

# Display API status
st.sidebar.markdown("<p class='sidebar-header'>API Status</p>", unsafe_allow_html=True)
if status["status"] == "online":
    st.sidebar.success(f"API Status: {status['status']}")
    st.sidebar.info(f"Models Loaded: {', '.join(status['models_loaded'])}")
    st.sidebar.info(f"Number of Users: {status['num_users']}")
    st.sidebar.info(f"Number of Songs: {status['num_songs']}")
else:
    st.sidebar.error(f"API Status: {status['status']}")

# Recommendation options
st.sidebar.markdown("<p class='sidebar-header'>Recommendation Options</p>", unsafe_allow_html=True)
num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 10)

# Tabs
tab1, tab2, tab3 = st.tabs(["User Recommendations", "Song Recommendations", "Feature-based Recommendations"])

# Function to display recommendations
def display_recommendations(recommendations: List[Dict[str, Any]]):
    if not recommendations:
        st.warning("No recommendations found.")
        return
    
    for i, rec in enumerate(recommendations):
        with st.container():
            st.markdown(f"""
            <div class='recommendation-card'>
                <div class='song-title'>{i+1}. {rec['song_name']}</div>
                <div class='artist-name'>Artist: {rec['artist_name']}</div>
                <div class='album-name'>Album: {rec['album_name']}</div>
                <div><span class='genre-tag'>{rec['genre']}</span></div>
                <div class='feature-value'>Popularity: {rec.get('popularity', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show audio features in an expander
            with st.expander("Audio Features"):
                features = {k: v for k, v in rec.items() if k in config.AUDIO_FEATURES and v is not None}
                if features:
                    # Create a bar chart for the features
                    feature_df = pd.DataFrame({
                        'Feature': list(features.keys()),
                        'Value': list(features.values())
                    })
                    st.bar_chart(feature_df.set_index('Feature'))
                else:
                    st.info("No audio features available for this song.")

# User Recommendations Tab
with tab1:
    st.markdown("### Get Recommendations for a User")
    st.write("Enter a user ID to get personalized recommendations.")
    
    user_id = st.text_input("User ID", "0")
    exclude_listened = st.checkbox("Exclude songs the user has already listened to", value=True)
    
    if st.button("Get User Recommendations"):
        try:
            response = requests.post(
                f"{API_URL}/recommend/user",
                json={
                    "user_id": user_id,
                    "num_recommendations": num_recommendations,
                    "exclude_listened": exclude_listened
                },
                params={"model_name": model_name}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"Recommendations generated using {data['model_used']} model")
                display_recommendations(data["recommendations"])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Song Recommendations Tab
with tab2:
    st.markdown("### Get Similar Song Recommendations")
    st.write("Enter a song ID to find similar songs.")
    
    song_id = st.text_input("Song ID", "song_0")
    
    if st.button("Get Similar Songs"):
        try:
            response = requests.post(
                f"{API_URL}/recommend/song",
                json={
                    "song_id": song_id,
                    "num_recommendations": num_recommendations
                },
                params={"model_name": model_name}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"Recommendations generated using {data['model_used']} model")
                display_recommendations(data["recommendations"])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Feature-based Recommendations Tab
with tab3:
    st.markdown("### Get Recommendations Based on Features")
    st.write("Specify audio features to get recommendations for new users (cold-start).")
    st.info("This feature is only available with the GraphSAGE model due to its inductive learning capabilities.")
    
    # Create sliders for each feature
    features = {}
    col1, col2 = st.columns(2)
    
    with col1:
        features["danceability"] = st.slider("Danceability", 0.0, 1.0, 0.5)
        features["energy"] = st.slider("Energy", 0.0, 1.0, 0.5)
        features["speechiness"] = st.slider("Speechiness", 0.0, 1.0, 0.1)
        features["acousticness"] = st.slider("Acousticness", 0.0, 1.0, 0.3)
        features["instrumentalness"] = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    
    with col2:
        features["liveness"] = st.slider("Liveness", 0.0, 1.0, 0.2)
        features["valence"] = st.slider("Valence (Positivity)", 0.0, 1.0, 0.5)
        features["tempo"] = st.slider("Tempo", 60.0, 200.0, 120.0)
        features["loudness"] = st.slider("Loudness", -20.0, 0.0, -8.0)
    
    if st.button("Get Feature-Based Recommendations"):
        try:
            response = requests.post(
                f"{API_URL}/recommend/features",
                json={
                    "features": features,
                    "num_recommendations": num_recommendations
                },
                params={"model_name": "graphsage"}  # Force GraphSAGE for feature-based recommendations
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"Recommendations generated using {data['model_used']} model")
                display_recommendations(data["recommendations"])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### About TunedIn")
st.write("""
TunedIn is an advanced AI music recommendation system that personalizes song suggestions using Graph Neural Networks (GNNs), 
user listening habits, and audio feature analysis to create a seamless and intelligent music discovery experience.
""")

st.markdown("### How it Works")
st.write("""
1. **Graph Neural Networks**: We represent users, songs, artists, and genres as a graph and use GNNs to learn patterns.
2. **Multiple Models**: We support various GNN architectures (GraphSAGE, GCN, GAT, LightGCN) for different recommendation scenarios.
3. **Cold-Start Handling**: Our GraphSAGE model can generate recommendations for new users based on their preferences.
4. **Audio Feature Analysis**: We incorporate audio features like tempo, energy, and danceability for better recommendations.
""")

# Run the app
if __name__ == "__main__":
    pass
    # To run the app, use the following command:
    # streamlit run web/app.py 