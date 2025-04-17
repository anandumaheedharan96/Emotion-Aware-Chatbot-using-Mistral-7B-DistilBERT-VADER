from flask import Flask, request, jsonify, render_template
import time
import requests
import random
import torch
from llama_cpp import Llama
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
import spacy
import os

app = Flask(__name__)

# Path to your model file
model_path = r"C:/Users/HP/.lmstudio/models/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

# Initialize the model
llm = Llama(model_path=model_path, n_ctx=4096, verbose=True)

# API Keys
YOUTUBE_API_KEY = "AIzaSyC7ctqYNrAPqEddNW2AMw2TBZfBG7_utQY"
SPOTIFY_CLIENT_ID = "a456729bd80f4fa582137bd4e6537c25"
SPOTIFY_CLIENT_SECRET = "b436aa8eb1a34d1d96ec909a54941af1"
RAPIDAPI_KEY = "your_rapidapi_key"  # Replace with your RapidAPI key

# Base URLs for APIs
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
TRIVIA_API_URL = "https://opentdb.com/api.php"
SUDOKU_API_URL = "https://sudoku-api.vercel.app/api/dosuku"  # Sudoku API
CROSSWORD_API_URL = "https://1-clickcrossword.promptsea.io/api/generate"  # Crossword API

# Load NLP models
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
vader_analyzer = SentimentIntensityAnalyzer()

# Store session data
conversation_histories = {}
spotify_tokens = {}

# Fetch Spotify access token
def get_spotify_token():
    response = requests.post(
        SPOTIFY_TOKEN_URL,
        {
            "grant_type": "client_credentials",
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        },
    )
    token = response.json().get("access_token")
    return token

# Fetch songs from Spotify
def fetch_spotify_songs(query, token, max_results=3):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": max_results}
    response = requests.get(SPOTIFY_SEARCH_URL, headers=headers, params=params)
    tracks = response.json().get("tracks", {}).get("items", [])

    track_links = []
    for track in tracks:
        track_url = track.get("external_urls", {}).get("spotify")
        if track_url:
            track_links.append(track_url)

    return track_links if track_links else ["No songs found."]

# Fetch videos from YouTube
def fetch_youtube_videos(query, max_results=3):
    params = {
        "part": "snippet",
        "q": query,
        "key": YOUTUBE_API_KEY,
        "type": "video",
        "maxResults": max_results,
    }
    response = requests.get(YOUTUBE_SEARCH_URL, params=params)
    videos = response.json().get("items", [])

    video_links = []
    for video in videos:
        video_id = video.get("id", {}).get("videoId")
        if video_id:
            video_links.append(f"https://www.youtube.com/watch?v={video_id}")

    return video_links if video_links else ["No videos found."]

# Enhanced emotion detection function using DistilBERT and VADER
def detect_emotion(user_input):
    # Get VADER sentiment scores
    vader_scores = vader_analyzer.polarity_scores(user_input)
    
    # Get DistilBERT emotion classification
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # Get probabilities for each emotion class
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    
    # Map probability indices to emotion labels
    emotions = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
    emotion_scores = {emotion: prob for emotion, prob in zip(emotions, probs)}
    
    # Determine primary emotion from DistilBERT
    primary_emotion = emotions[probs.index(max(probs))]
    
    # Adjust based on VADER scores
    compound_score = vader_scores['compound']
    
    # Logic for combining models
    if compound_score >= 0.5 and primary_emotion not in ["Joy", "Surprise"]:
        if emotion_scores["Joy"] > 0.15:  # If joy score is reasonable
            primary_emotion = "Joy"
    elif compound_score <= -0.5 and primary_emotion not in ["Anger", "Disgust", "Sadness", "Fear"]:
        # Choose the highest negative emotion
        neg_emotions = {"Anger": emotion_scores["Anger"], 
                        "Disgust": emotion_scores["Disgust"],
                        "Sadness": emotion_scores["Sadness"],
                        "Fear": emotion_scores["Fear"]}
        primary_emotion = max(neg_emotions, key=neg_emotions.get)
    
    return primary_emotion, emotion_scores, vader_scores

# Fetch Sudoku puzzle
def fetch_sudoku_puzzle():
    try:
        response = requests.get(SUDOKU_API_URL)
        if response.status_code == 200:
            puzzle_data = response.json()
            puzzle = puzzle_data.get("newboard", {}).get("grids", [])[0].get("value", [])
            solution = puzzle_data.get("newboard", {}).get("grids", [])[0].get("solution", [])
            if puzzle and solution:
                return {"puzzle": puzzle, "solution": solution}
            else:
                return {"error": "Failed to fetch a Sudoku puzzle. The board data is empty."}
        else:
            return {"error": "Failed to fetch a Sudoku puzzle. API response error."}
    except Exception as e:
        return {"error": f"Error fetching Sudoku puzzle: {str(e)}"}

# Fetch a trivia question based on the topic
def fetch_trivia_question(topic):
    # Map user topics to Open Trivia Database categories
    category_map = {
        "science": 17,
        "history": 23,
        "math": 19,
        "mathematics": 19,
        "technology": 18,
        "physics": 17,  # Science & Nature category
        "chemistry": 17,  # Science & Nature category
        "biology": 17,  # Science & Nature category
        "geography": 22,
        "art": 25,
        "sports": 21,
        "general": 9,
    }
    # Convert topic to lowercase and trim whitespace
    topic = topic.lower().strip()
    category_id = category_map.get(topic, 9)  # Default to General Knowledge if topic not found

    params = {
        "amount": 1,
        "type": "multiple",
        "category": category_id,
    }
    
    try:
        response = requests.get(TRIVIA_API_URL, params=params)
        data = response.json()
        if data.get("results"):
            question_data = data["results"][0]
            question = question_data["question"]
            correct_answer = question_data["correct_answer"]
            options = question_data["incorrect_answers"] + [correct_answer]
            random.shuffle(options)
            
            return {
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "topic": topic.capitalize()
            }
        else:
            return {"error": "Sorry, no trivia questions found for this topic."}
    except Exception as e:
        return {"error": f"Error fetching trivia question: {str(e)}"}

# Get quiz topics
def get_quiz_topics():
    return [
        "Science",
        "History",
        "Math",
        "Technology",
        "Physics",
        "Chemistry",
        "Biology",
        "Geography",
        "Art",
        "Sports",
        "General Knowledge",
    ]

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    # Initialize session if not exists
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    if session_id not in spotify_tokens:
        spotify_tokens[session_id] = get_spotify_token()
    
    # Process user input
    doc = nlp(user_input)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    
    # Store keywords in history
    conversation_histories[session_id].append(" ".join(keywords))
    if len(conversation_histories[session_id]) > 3:
        conversation_histories[session_id].pop(0)
    
    # Chat with LLM
    prompt = f"[INST] {user_input} [/INST]"
    response = llm(prompt, max_tokens=256, temperature=0.7, top_p=0.9)
    ai_response = response["choices"][0]["text"].strip()
    
    return jsonify({"response": ai_response})

@app.route('/api/study', methods=['GET'])
def study_options():
    return jsonify({"message": "What do you feel like doing today?", "options": ["videos", "quiz"]})

@app.route('/api/videos', methods=['POST'])
def get_videos():
    data = request.json
    topic = data.get('topic', '')
    sub_topic = data.get('sub_topic', '')
    
    query = f"{topic} {sub_topic} study tutorial" if sub_topic else f"{topic} study tutorial"
    try:
        videos = fetch_youtube_videos(query)
        return jsonify({"videos": videos})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/quiz/topics', methods=['GET'])
def quiz_topics():
    return jsonify({"topics": get_quiz_topics()})

@app.route('/api/quiz', methods=['POST'])
def get_quiz():
    data = request.json
    topic = data.get('topic', 'general')
    return jsonify(fetch_trivia_question(topic))

@app.route('/api/verify_answer', methods=['POST'])
def verify_answer():
    data = request.json
    user_answer = data.get('answer', '')
    correct_answer = data.get('correct_answer', '')
    is_correct = user_answer == correct_answer
    
    return jsonify({
        "is_correct": is_correct,
        "correct_answer": correct_answer
    })

@app.route('/api/suggest_video', methods=['POST'])
def suggest_video():
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id not in conversation_histories:
        return jsonify({"error": "Session not found"}), 404
    
    combined_input = " ".join(conversation_histories[session_id])
    emotion, _, _ = detect_emotion(combined_input)
    
    try:
        query = emotion + " " + combined_input
        videos = fetch_youtube_videos(query)
        return jsonify({
            "videos": videos,
            "emotion": emotion
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/suggest_song', methods=['POST'])
def suggest_song():
    data = request.json
    session_id = data.get('session_id', 'default')
    genre = data.get('genre', '')
    
    if session_id not in conversation_histories:
        return jsonify({"error": "Session not found"}), 404
    
    if session_id not in spotify_tokens:
        spotify_tokens[session_id] = get_spotify_token()
    
    combined_input = " ".join(conversation_histories[session_id])
    emotion, _, _ = detect_emotion(combined_input)
    
    query = genre if genre else emotion + " " + combined_input
    
    try:
        songs = fetch_spotify_songs(query, spotify_tokens[session_id])
        return jsonify({
            "songs": songs,
            "emotion": emotion
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sudoku', methods=['GET'])
def get_sudoku():
    return jsonify(fetch_sudoku_puzzle())

@app.route('/api/emotion', methods=['POST'])
def get_emotion():
    data = request.json
    text = data.get('text', '')
    
    emotion, emotion_scores, vader_scores = detect_emotion(text)
    
    return jsonify({
        "primary_emotion": emotion,
        "emotion_scores": emotion_scores,
        "vader_scores": vader_scores
    })

if __name__ == '__main__':
    app.run(debug=True)