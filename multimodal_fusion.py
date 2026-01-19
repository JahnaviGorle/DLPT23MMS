import logging
from collections import Counter

logger = logging.getLogger(__name__)

class MultimodalFusion:
    """
    Combines results from text, video, and audio emotion detection
    """
    
    # Emotion mapping between different modalities
    EMOTION_MAPPING = {
        # Audio emotions: neutral, calm, happy, sad, angry, fear, disgust, surprise
        # Video emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
        # Text emotions: happy, sad, angry, fear, disgust, surprise, neutral, calm
        
        'Angry': 'angry',
        'angry': 'angry',
        'Disgust': 'disgust',
        'disgust': 'disgust',
        'disgusted': 'disgust',
        'Fear': 'fear',
        'fear': 'fear',
        'fearful': 'fear',
        'Happy': 'happy',
        'happy': 'happy',
        'Neutral': 'neutral',
        'neutral': 'neutral',
        'Sad': 'sad',
        'sad': 'sad',
        'Surprise': 'surprise',
        'surprise': 'surprise',
        'surprised': 'surprise',
        'calm': 'calm',
    }
    
    def normalize_emotion(self, emotion):
        """Normalize emotion name to standard format"""
        emotion_lower = emotion.lower().strip()
        return self.EMOTION_MAPPING.get(emotion, self.EMOTION_MAPPING.get(emotion_lower, emotion_lower))
    
    def fuse_results(self, text_result, video_result, audio_result, weights=None):
        """
        Fuse results from three modalities
        
        Args:
            text_result: dict with 'dominant_emotion' and 'confidence_scores'
            video_result: dict with 'dominant_emotion' and 'emotion_distribution'
            audio_result: dict with 'dominant_emotion' and 'confidence_scores'
            weights: dict with weights for each modality (default: equal weights)
        
        Returns:
            Combined emotion result
        """
        try:
            if weights is None:
                weights = {'text': 0.3, 'video': 0.4, 'audio': 0.3}
            
            # Normalize emotions
            text_emotion = self.normalize_emotion(text_result.get('dominant_emotion', 'neutral'))
            video_emotion = self.normalize_emotion(video_result.get('dominant_emotion', 'neutral'))
            audio_emotion = self.normalize_emotion(audio_result.get('dominant_emotion', 'neutral'))
            
            # Get confidence scores
            text_confidence = self._get_confidence(text_result, text_emotion)
            video_confidence = self._get_video_confidence(video_result, video_emotion)
            audio_confidence = self._get_confidence(audio_result, audio_emotion)
            
            # Weighted voting
            emotion_scores = {}
            all_emotions = set([text_emotion, video_emotion, audio_emotion])
            
            for emotion in all_emotions:
                score = 0.0
                
                if emotion == text_emotion:
                    score += text_confidence * weights['text']
                if emotion == video_emotion:
                    score += video_confidence * weights['video']
                if emotion == audio_emotion:
                    score += audio_confidence * weights['audio']
                
                emotion_scores[emotion] = score
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            dominant_score = emotion_scores[dominant_emotion]
            
            # Normalize scores to percentages
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: (v / total_score) * 100 for k, v in emotion_scores.items()}
            
            logger.info(f"✅ Fused emotion: {dominant_emotion} (score: {dominant_score:.2f})")
            
            return {
                'dominant_emotion': dominant_emotion,
                'confidence': round(dominant_score * 100, 2),
                'emotion_scores': emotion_scores,
                'modality_results': {
                    'text': {
                        'emotion': text_emotion,
                        'confidence': round(text_confidence, 2)
                    },
                    'video': {
                        'emotion': video_emotion,
                        'confidence': round(video_confidence, 2)
                    },
                    'audio': {
                        'emotion': audio_emotion,
                        'confidence': round(audio_confidence, 2)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error fusing results: {str(e)}")
            # Fallback to simple majority vote
            emotions = [
                self.normalize_emotion(text_result.get('dominant_emotion', 'neutral')),
                self.normalize_emotion(video_result.get('dominant_emotion', 'neutral')),
                self.normalize_emotion(audio_result.get('dominant_emotion', 'neutral'))
            ]
            dominant_emotion = Counter(emotions).most_common(1)[0][0]
            
            return {
                'dominant_emotion': dominant_emotion,
                'confidence': 50.0,
                'emotion_scores': {dominant_emotion: 100.0},
                'modality_results': {
                    'text': {'emotion': emotions[0], 'confidence': 33.3},
                    'video': {'emotion': emotions[1], 'confidence': 33.3},
                    'audio': {'emotion': emotions[2], 'confidence': 33.3}
                }
            }
    
    def _get_confidence(self, result, emotion):
        """Extract confidence score for a specific emotion"""
        confidence_scores = result.get('confidence_scores', {})
        if emotion in confidence_scores:
            return confidence_scores[emotion] / 100.0
        return 0.5  # Default confidence
    
    def _get_video_confidence(self, result, emotion):
        """Extract confidence from video result"""
        emotion_dist = result.get('emotion_distribution', {})
        total = result.get('total_frames_processed', 1)
        
        # Normalize emotion name for matching
        for key, value in emotion_dist.items():
            if self.normalize_emotion(key) == emotion:
                return value / total if total > 0 else 0.0
        
        return 0.5  # Default confidence

