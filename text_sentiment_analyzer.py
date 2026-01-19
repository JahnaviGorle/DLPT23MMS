import os
import google.genai as genai
from dotenv import load_dotenv
import logging
import json
import re

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TextSentimentAnalyzer:
    def __init__(self):
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            self.client = genai.Client(api_key=api_key)
            logger.info("✅ Text Sentiment Analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing TextSentimentAnalyzer: {str(e)}")
            raise
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment/emotion from text using Gemini
        Returns emotion classification
        """
        try:
            if not text or len(text.strip()) < 10:
                logger.warning("Text too short for analysis")
                return {
                    "dominant_emotion": "neutral",
                    "confidence_scores": {"neutral": 100.0},
                    "reasoning": "Insufficient text for analysis"
                }
            
            # Create prompt for emotion analysis
            prompt = f"""Analyze the sentiment and emotion of the following text from a YouTube video transcript/description.

Text: {text[:2000]}  # Limit to first 2000 chars

Based on the text content, classify the dominant emotion into one of these categories:
- happy
- sad
- angry
- fear
- disgust
- surprise
- neutral
- calm

Respond ONLY with a JSON object in this exact format:
{{
    "dominant_emotion": "emotion_name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Do not include any other text, only the JSON object."""

            # Call Gemini API
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )
            
            response_text = response.text.strip()
            
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            result = json.loads(response_text)
            
            # Normalize emotion name
            emotion = result.get("dominant_emotion", "neutral").lower()
            confidence = float(result.get("confidence", 0.5))
            
            # Create confidence scores for all emotions
            confidence_scores = {e: 0.0 for e in ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral", "calm"]}
            confidence_scores[emotion] = confidence * 100
            
            logger.info(f"✅ Text emotion detected: {emotion} (confidence: {confidence*100:.1f}%)")
            
            return {
                "dominant_emotion": emotion,
                "confidence_scores": confidence_scores,
                "reasoning": result.get("reasoning", "")
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            logger.error(f"Response was: {response_text}")
            return {
                "dominant_emotion": "neutral",
                "confidence_scores": {"neutral": 100.0},
                "reasoning": "Error parsing AI response"
            }
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return {
                "dominant_emotion": "neutral",
                "confidence_scores": {"neutral": 100.0},
                "reasoning": f"Error: {str(e)}"
            }

