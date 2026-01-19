
import cv2
import numpy as np
import pandas as pd
import time
from mtcnn import MTCNN
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import os
import logging

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        try:
            # Load the trained emotion detection model
            model_path = os.path.join(os.path.dirname(__file__), "model", "model.h5")
            logger.info(f"Loading model from: {model_path}")
            self.classifier = load_model(model_path)
            self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            self.detector = MTCNN()
        except Exception as e:
            logger.error(f"Error initializing EmotionDetector: {str(e)}")
            raise

    def preprocess_face(self, face):
        try:
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
            roi = gray_face.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            return roi
        except Exception as e:
            logger.error(f"Error processing face: {str(e)}")
            return None

    def process_video_from_path(self, video_path):
        """
        Process video from file path (for YouTube processing)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")

            frame_skip = 60  # Process every 5th frame
            frame_count = 0
            emotion_data = []
            emotions_with_scores = {}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb_frame)

                for face in faces:
                    x, y, w, h = face['box']
                    confidence = face['confidence']

                    if confidence < 0.95:
                        continue

                    face_roi = frame[y:y + h, x:x + w]
                    roi = self.preprocess_face(face_roi)

                    if roi is None:
                        continue

                    predictions = self.classifier.predict(roi)[0]
                    emotions_with_scores = {
                        self.emotion_labels[i]: float(predictions[i] * 100) 
                        for i in range(len(self.emotion_labels))
                    }
                    dominant_emotion = max(emotions_with_scores, key=emotions_with_scores.get)
                    confidence_score = emotions_with_scores[dominant_emotion]

                    emotion_data.append({
                        'frame': frame_count,
                        'emotion': dominant_emotion,
                        'confidence': round(confidence_score, 2)
                    })

            # Process results
            if emotion_data:
                # Convert to DataFrame for analysis
                df = pd.DataFrame(emotion_data)
                emotion_counts = df['emotion'].value_counts()
                most_frequent_emotion = emotion_counts.index[0]
                
                results = {
                    'dominant_emotion': most_frequent_emotion,
                    'occurrence_count': int(emotion_counts[most_frequent_emotion]),
                    'emotion_distribution': {
                        emotion: int(count) 
                        for emotion, count in emotion_counts.items()
                    },
                    'total_frames_processed': len(emotion_data)
                }
                
                logger.info(f"Video processing results: {results}")
                return results
            else:
                logger.warning("No faces detected in the video")
                raise Exception("No faces detected in the video")

        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            raise
        finally:
            if 'cap' in locals():
                cap.release()

    def process_video(self, video_file):
        # Save uploaded file temporarily
        temp_path = "temp_video.mp4"
        logger.info(f"Saving temporary video file to: {temp_path}")
        video_file.save(temp_path)
        
        try:
            result = self.process_video_from_path(temp_path)
            return result
        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Temporary video file removed")

def process_video(video_file):
    try:
        detector = EmotionDetector()
        return detector.process_video(video_file)
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        raise

