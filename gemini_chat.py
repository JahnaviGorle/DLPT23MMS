import os
import google.genai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Gemini client (NEW SDK way)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMOTIONS = {
    "Neutral": "Let's do something fun! Would you like to play a quick game, hear an interesting fact, or get a fun challenge?",
    "Happy": "You're in a great mood! Let's keep the positivity going. Want to listen to an uplifting story, get a fun challenge, or share a happy memory?",
    "Sad": "I'm here for you. Would you like some calming music recommendations, a motivational story, or a virtual hug?",
    "Angry": "I understand frustration can be tough. Want to try a quick breathing exercise, hear a calming story, or vent your thoughts?",
    "Fear": "You're not alone! Let's ease your mind. Want a comforting quote, a guided relaxation, or an inspiring success story?",
    "Disgust": "Let's distract your mind. Want to hear a joke, learn a new interesting fact, or play a quick word game?",
    "Surprise": "Surprises can be fun! Want to share what happened, hear about a surprising event, or play a quick reaction game?",
    "Calm": "A peaceful moment is precious. Would you like to try meditation, listen to soothing nature sounds, or explore a new mindfulness tip?"
}

MOOD_ALIASES = {
    "fearful": "fear",
    "fear": "fear",
    "disgusted": "disgust",
    "disgust": "disgust",
    "surprised": "surprise",
    "surprise": "surprise",
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "calm": "calm",
}

def normalize_mood(mood: str) -> str:
    key = mood.strip().lower()
    normalized = MOOD_ALIASES.get(key, key)
    return normalized.title()

def get_mood_recommendation(mood: str) -> str:
    normalized_mood = normalize_mood(mood)
    return EMOTIONS.get(
        normalized_mood,
        "I didn't recognize that mood. Can you describe how you're feeling?"
    )

def process_message(message: str) -> str:
    mood = normalize_mood(message)

    # Emotion-based shortcut (no API call)
    if mood in EMOTIONS:
        return get_mood_recommendation(mood)

    # Gemini API call (NEW SDK)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=message
    )

    return response.text
