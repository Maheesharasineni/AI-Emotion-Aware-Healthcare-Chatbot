from flask import Flask, render_template, Response, jsonify, request, session
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId
import json
from camera import VideoCamera
import threading
import time

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configure Gemini AI
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

# MongoDB Configuration
MONGO_URI = os.environ.get('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client.mental_health_chatbot

# Collections
users_collection = db.users
chats_collection = db.chats
emotions_collection = db.emotions
daily_checkins_collection = db.daily_checkins

# Global variables for emotion detection
current_emotion = "neutral"
emotion_confidence = 0.0
camera_active = False

class EmotionBasedResponses:
    def __init__(self):
        self.emotion_prompts = {
            "angry": """You are a compassionate mental health assistant. The user is feeling angry. 
                       Provide empathetic support, validate their feelings, and suggest healthy coping strategies 
                       like deep breathing, physical exercise, or talking to someone. Keep responses supportive and non-judgmental.""",
            
            "sad": """You are a caring mental health assistant. The user is feeling sad or down. 
                     Offer gentle encouragement, validate their emotions, and suggest activities that might help 
                     like journaling, connecting with loved ones, or engaging in self-care. Be warm and understanding.""",
            "fearful": """You are a supportive mental health assistant. The user is feeling anxious or fearful. 
                         Provide calming techniques, grounding exercises, and reassurance. Suggest breathing exercises, 
                         mindfulness, or breaking down overwhelming situations into manageable steps.""",
            
            "happy": """You are an encouraging mental health assistant. The user is feeling happy or positive. 
                       Celebrate their positive emotions, help them reflect on what's going well, and suggest ways 
                       to maintain this positive state. Encourage gratitude practices and positive activities.""",
            
            "neutral": """You are a helpful mental health assistant. The user seems neutral or calm. 
                         Offer general mental wellness tips, ask about their day, and provide supportive conversation. 
                         Be ready to adapt based on what they share.""",
            
            "surprised": """You are an understanding mental health assistant. The user seems surprised or startled. 
                           Help them process unexpected emotions or situations, offer grounding techniques, and 
                           provide reassurance if needed.""",
            
            "disgusted": """You are a patient mental health assistant. The user seems frustrated or disgusted. 
                           Help them work through difficult feelings, validate their experience, and suggest 
                           healthy ways to process these emotions."""
        }

    def get_response(self, user_message, emotion, chat_history=[]):
        prompt = self.emotion_prompts.get(emotion.lower(), self.emotion_prompts["neutral"])
        
        context = f"{prompt}\n\nUser's current emotion: {emotion}\n"
        if chat_history:
            context += "Recent conversation:\n"
            for chat in chat_history[-3:]:  # Last 3 messages for context
                context += f"User: {chat.get('user_message', '')}\nAssistant: {chat.get('bot_response', '')}\n"
        
        context += f"\nUser's current message: {user_message}\n\nProvide a helpful, empathetic response (max 200 words):"
        
        try:
            response = model.generate_content(context)
            return response.text
        except Exception as e:
            return self._get_fallback_response(emotion)
    
    def _get_fallback_response(self, emotion):
        fallbacks = {
            "angry": "I understand you're feeling angry right now. That's completely valid. Try taking some deep breaths - in for 4 counts, hold for 4, out for 4. Would you like to talk about what's bothering you?",
            "sad": "I'm here for you during this difficult time. Your feelings are valid and it's okay to feel sad. Sometimes talking helps - I'm listening. Would you like to share what's on your mind?",
            "fearful": "I can sense you're feeling anxious or worried. You're safe right now. Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
            "happy": "It's wonderful to see you feeling positive! What's bringing you joy today? Celebrating good moments is so important for our mental wellbeing.",
            "neutral": "How are you feeling today? I'm here to listen and support you with whatever you'd like to talk about.",
            "surprised": "You seem surprised or caught off guard. Take a moment to breathe and process. I'm here if you need to talk through anything.",
            "disgusted": "I can see you're feeling frustrated or upset about something. Those feelings are valid. Would you like to talk about what's bothering you?"
        }
        return fallbacks.get(emotion.lower(), fallbacks["neutral"])

emotion_responder = EmotionBasedResponses()

def get_emergency_resources():
    return {
        "crisis_hotlines": [
            {"name": "National Suicide Prevention Lifeline", "number": "988", "available": "24/7"},
            {"name": "Crisis Text Line", "number": "Text HOME to 741741", "available": "24/7"},
            {"name": "SAMHSA National Helpline", "number": "1-800-662-4357", "available": "24/7"}
        ],
        "immediate_actions": [
            "Reach out to a trusted friend or family member",
            "Contact your therapist or counselor if you have one",
            "Visit your nearest emergency room if in immediate danger",
            "Call emergency services (911) if you're having thoughts of self-harm"
        ]
    }

def assess_emotional_risk(emotion, user_message):
    """Assess if user needs immediate help based on emotion and message content"""
    risk_keywords = ['suicide', 'kill myself', 'end it all', 'hurt myself', 'self-harm', 'worthless', 'hopeless']
    high_risk_emotions = ['angry', 'sad', 'fearful']
    
    message_lower = user_message.lower()
    has_risk_keywords = any(keyword in message_lower for keyword in risk_keywords)
    is_high_risk_emotion = emotion.lower() in high_risk_emotions
    
    if has_risk_keywords:
        return "high"
    elif is_high_risk_emotion and len(user_message) > 50:  # Longer messages with negative emotions
        return "medium"
    else:
        return "low"

@app.route('/')
def index():
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('chatbot.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Check if user exists
        if users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
            return jsonify({'success': False, 'message': 'User already exists'})
        
        # Create new user
        user_data = {
            'username': username,
            'email': email,
            'password': generate_password_hash(password),
            'created_at': datetime.utcnow(),
            'privacy_consent': data.get('privacy_consent', False),
            'camera_consent': data.get('camera_consent', False)
        }
        
        result = users_collection.insert_one(user_data)
        session['user_id'] = str(result.inserted_id)
        session['username'] = username
        
        return jsonify({'success': True, 'message': 'Registration successful'})
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = username
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'})
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return render_template('login.html')

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    user_message = data.get('message', '')
    manual_emotion = data.get('emotion', None)
    
    # Use manual emotion if provided, otherwise use detected emotion
    emotion = manual_emotion if manual_emotion else current_emotion
    
    # Get recent chat history for context
    recent_chats = list(chats_collection.find(
        {'user_id': session['user_id']},
        sort=[('timestamp', -1)],
        limit=5
    ))
    
    # Generate AI response
    bot_response = emotion_responder.get_response(user_message, emotion, recent_chats)
    
    # Assess emotional risk
    risk_level = assess_emotional_risk(emotion, user_message)
    
    # Save chat to database
    chat_data = {
        'user_id': session['user_id'],
        'user_message': user_message,
        'bot_response': bot_response,
        'emotion': emotion,
        'emotion_confidence': emotion_confidence,
        'risk_level': risk_level,
        'timestamp': datetime.utcnow()
    }
    chats_collection.insert_one(chat_data)
    
    # Log emotion data
    emotion_data = {
        'user_id': session['user_id'],
        'emotion': emotion,
        'confidence': emotion_confidence,
        'source': 'manual' if manual_emotion else 'camera',
        'timestamp': datetime.utcnow()
    }
    emotions_collection.insert_one(emotion_data)
    
    response_data = {
        'response': bot_response,
        'emotion': emotion,
        'risk_level': risk_level
    }
    
    if risk_level == 'high':
        response_data['emergency_resources'] = get_emergency_resources()
    
    return jsonify(response_data)

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return "Unauthorized", 401
    
    def generate():
        global current_emotion, emotion_confidence, camera_active
        camera = VideoCamera()
        camera_active = True
        
        while camera_active:
            try:
                frame, emotion_data = camera.get_frame()
                if emotion_data:
                    current_emotion = emotion_data['emotion']
                    emotion_confidence = emotion_data['confidence']
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            except Exception as e:
                print(f"Video feed error: {e}")
                break
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_data')
def get_emotion_data():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    return jsonify({
        'emotion': current_emotion,
        'confidence': emotion_confidence,
        'camera_active': camera_active
    })

@app.route('/daily_checkin', methods=['GET', 'POST'])
def daily_checkin():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'POST':
        data = request.get_json()
        
        # Check if user already checked in today
        today = datetime.utcnow().date()
        existing_checkin = daily_checkins_collection.find_one({
            'user_id': session['user_id'],
            'date': today
        })
        
        if existing_checkin:
            return jsonify({'success': False, 'message': 'Already checked in today'})
        
        checkin_data = {
            'user_id': session['user_id'],
            'date': today,
            'mood_rating': data.get('mood_rating'),
            'sleep_hours': data.get('sleep_hours'),
            'stress_level': data.get('stress_level'),
            'activities': data.get('activities', []),
            'notes': data.get('notes', ''),
            'timestamp': datetime.utcnow()
        }
        
        daily_checkins_collection.insert_one(checkin_data)
        return jsonify({'success': True, 'message': 'Check-in saved successfully'})
    
    # GET request - return today's check-in if exists
    today = datetime.utcnow().date()
    checkin = daily_checkins_collection.find_one({
        'user_id': session['user_id'],
        'date': today
    })
    
    if checkin:
        checkin['_id'] = str(checkin['_id'])
        return jsonify(checkin)
    else:
        return jsonify({'checked_in': False})

@app.route('/mood_history')
def mood_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Get last 30 days of emotion data
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    emotions = list(emotions_collection.find({
        'user_id': session['user_id'],
        'timestamp': {'$gte': thirty_days_ago}
    }).sort('timestamp', 1))
    
    # Process data for charting
    mood_data = {}
    for emotion in emotions:
        date = emotion['timestamp'].date().isoformat()
        if date not in mood_data:
            mood_data[date] = []
        mood_data[date].append(emotion['emotion'])
    
    # Calculate dominant emotion per day
    daily_moods = []
    for date, day_emotions in mood_data.items():
        dominant_emotion = max(set(day_emotions), key=day_emotions.count)
        daily_moods.append({
            'date': date,
            'emotion': dominant_emotion,
            'count': len(day_emotions)
        })
    
    return jsonify(daily_moods)

@app.route('/coping_strategies')
def coping_strategies():
    strategies = {
        'angry': [
            'Practice deep breathing exercises (4-7-8 technique)',
            'Try progressive muscle relaxation',
            'Go for a brisk walk or do physical exercise',
            'Write in a journal to express your feelings',
            'Listen to calming music',
            'Talk to a trusted friend or counselor'
        ],
        'sad': [
            'Practice gratitude by listing 3 good things from your day',
            'Engage in a creative activity you enjoy',
            'Reach out to supportive friends or family',
            'Try gentle exercise like yoga or walking',
            'Practice self-compassion and positive self-talk',
            'Consider professional counseling if sadness persists'
        ],
        'fearful': [
            'Use the 5-4-3-2-1 grounding technique',
            'Practice mindfulness meditation',
            'Challenge negative thoughts with facts',
            'Break overwhelming tasks into smaller steps',
            'Use positive affirmations',
            'Try belly breathing exercises'
        ],
        'happy': [
            'Practice gratitude to maintain positive feelings',
            'Share your joy with others',
            'Engage in activities that bring you fulfillment',
            'Help others through volunteer work',
            'Set new positive goals',
            'Take time to appreciate the present moment'
        ],
        'neutral': [
            'Set small, achievable goals for the day',
            'Try a new hobby or activity',
            'Practice mindfulness or meditation',
            'Connect with friends or family',
            'Engage in physical activity',
            'Practice good sleep hygiene'
        ]
    }
    
    emotion = request.args.get('emotion', 'neutral')
    return jsonify({
        'emotion': emotion,
        'strategies': strategies.get(emotion.lower(), strategies['neutral'])
    })

@app.route('/privacy_settings', methods=['GET', 'POST'])
def privacy_settings():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'POST':
        data = request.get_json()
        
        users_collection.update_one(
            {'_id': ObjectId(session['user_id'])},
            {'$set': {
                'privacy_consent': data.get('privacy_consent', False),
                'camera_consent': data.get('camera_consent', False),
                'data_retention_days': data.get('data_retention_days', 30),
                'updated_at': datetime.utcnow()
            }}
        )
        
        return jsonify({'success': True, 'message': 'Privacy settings updated'})
    
    # GET request
    user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    return jsonify({
        'privacy_consent': user.get('privacy_consent', False),
        'camera_consent': user.get('camera_consent', False),
        'data_retention_days': user.get('data_retention_days', 30)
    })

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'success': True})

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)