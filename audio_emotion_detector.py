
import tensorflow as tf
import librosa
import numpy as np
import os
import logging
import soundfile as sf

logger = logging.getLogger(__name__)

# Define constants
MODELS_DIR = os.path.join(os.path.dirname(__file__), "MODELS")
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "RECORDINGS")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

class AudioEmotionDetector:
    def __init__(self):
        try:
            model_path = os.path.join(MODELS_DIR, "model_bi-lstm.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model file is placed in the MODELS directory.")
            
            # Load model - Keras 3.0+ natively supports .keras format and Functional class
            try:
                import keras
                # Load with safe_mode=False to allow Functional class and other Keras 3 serialization
                self.model = keras.models.load_model(model_path, compile=False, safe_mode=False)
                logger.info("✅ Model loaded successfully using Keras 3!")
            except Exception as load_error:
                logger.error(f"Failed to load model: {str(load_error)[:300]}")
                raise FileNotFoundError(
                    f"Could not load model at {model_path}. "
                    f"Error: {str(load_error)[:300]}"
                )
            
            # Define emotion labels exactly as in original code
            self.emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]
            
        except Exception as e:
            logger.error(f"Error loading audio model: {str(e)}")
            raise

    def extract_features(self, audio_path):
        try:
            logger.info(f"Loading audio file from: {audio_path}")
            
            # First try with soundfile
            try:
                data, sample_rate = sf.read(audio_path)
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                # Trim to duration
                if len(data) > sample_rate * 2.5:
                    data = data[int(sample_rate * 0.6):int(sample_rate * 3.1)]  # 2.5s duration with 0.6s offset
            except Exception as sf_error:
                logger.warning(f"SoundFile failed, trying librosa: {sf_error}")
                data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
            
            logger.info(f"Audio loaded successfully. Shape: {data.shape}, Sample rate: {sample_rate}")
            
            # Extract 20 MFCCs exactly as in original code
            mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20)
            logger.info(f"MFCC features extracted. Shape before padding: {mfcc.shape}")

            # Handle padding/truncating exactly as in original code
            if mfcc.shape[1] < 108:
                pad_width = 108 - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            elif mfcc.shape[1] > 108:
                mfcc = mfcc[:, :108]
            
            logger.info(f"MFCC features padded/truncated. Final shape: {mfcc.shape}")

            # Add batch dimension as in original code
            mfcc = np.expand_dims(mfcc, axis=0)
            return mfcc
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def predict_emotion_from_path(self, audio_path):
        """
        Predict emotion from audio file path (for YouTube processing)
        """
        try:
            # Extract features using the same method as original code
            features = self.extract_features(audio_path)
            logger.info(f"Features extracted successfully. Shape: {features.shape}")
            
            # Make prediction exactly as in original code
            prediction = self.model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction)
            predicted_emotion = self.emotion_labels[predicted_class]
            
            # Calculate confidence scores
            confidence_scores = {
                emotion: float(score) * 100 
                for emotion, score in zip(self.emotion_labels, prediction[0])
            }
            
            logger.info(f"🎭 Predicted Emotion: {predicted_emotion}")
            
            return {
                "dominant_emotion": predicted_emotion,
                "confidence_scores": confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion from path: {str(e)}")
            raise

    def predict_emotion(self, audio_file):
        try:
            # Save the audio file with proper extension
            audio_path = os.path.join(RECORDINGS_DIR, "temp_recording.wav")
            logger.info(f"Saving audio file to: {audio_path}")
            
            audio_file.save(audio_path)
            logger.info("Audio file saved successfully")
            
            try:
                result = self.predict_emotion_from_path(audio_path)
                return result
                
            except Exception as pred_error:
                logger.error(f"Prediction error: {str(pred_error)}")
                raise
            finally:
                # Clean up temporary file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info("Temporary audio file cleaned up")
                
        except Exception as e:
            logger.error(f"Error predicting emotion: {str(e)}")
            raise

def process_audio(audio_file):
    try:
        detector = AudioEmotionDetector()
        return detector.predict_emotion(audio_file)
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        raise
