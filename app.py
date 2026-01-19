from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import traceback

from youtube_processor import YouTubeProcessor
from audio_emotion_detector import AudioEmotionDetector
from emotion_detector import EmotionDetector
from text_sentiment_analyzer import TextSentimentAnalyzer
from multimodal_fusion import MultimodalFusion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Initialize processors (lazy loading)
youtube_processor = None
audio_detector = None
video_detector = None
text_analyzer = None
fusion_engine = None

def initialize_processors():
    """Initialize all processors (called on first request)"""
    global youtube_processor, audio_detector, video_detector, text_analyzer, fusion_engine
    
    if youtube_processor is None:
        logger.info("Initializing processors...")
        youtube_processor = YouTubeProcessor()
        audio_detector = AudioEmotionDetector()
        video_detector = EmotionDetector()
        text_analyzer = TextSentimentAnalyzer()
        fusion_engine = MultimodalFusion()
        logger.info("✅ All processors initialized")

@app.route('/')
def index():
    """Serve the HTML interface"""
    return app.send_static_file('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Multimodal Sentiment Analysis API is running'
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_youtube_video():
    """
    Main endpoint for analyzing YouTube videos
    Expected JSON: {"youtube_url": "https://youtube.com/watch?v=..."}
    """
    try:
        initialize_processors()
        
        # Get YouTube URL from request
        data = request.get_json()
        if not data or 'youtube_url' not in data:
            return jsonify({
                'error': 'Missing youtube_url in request body'
            }), 400
        
        youtube_url = data['youtube_url']
        logger.info(f"Processing YouTube URL: {youtube_url}")
        
        video_id = None
        try:
            # Step 1: Download video and extract audio
            logger.info("Step 1: Downloading video and extracting audio...")
            download_info = youtube_processor.download_video(youtube_url)
            video_path = download_info['video_path']
            audio_path = download_info['audio_path']
            video_id = download_info['video_id']
            
        except Exception as download_error:
            logger.error(f"Error downloading video: {str(download_error)}")
            raise
        
        try:
            # Step 2: Extract transcript
            logger.info("Step 2: Extracting transcript...")
            transcript = youtube_processor.get_transcript(youtube_url)
            if not transcript:
                transcript = download_info.get('description', '')
            
            # Step 3: Analyze text sentiment
            logger.info("Step 3: Analyzing text sentiment...")
            text_result = text_analyzer.analyze_sentiment(transcript)
            
            # Step 4: Analyze video (facial emotions)
            logger.info("Step 4: Analyzing video frames...")
            video_result = video_detector.process_video_from_path(video_path)
            
            # Step 5: Analyze audio
            logger.info("Step 5: Analyzing audio...")
            audio_result = audio_detector.predict_emotion_from_path(audio_path)
            
            # Step 6: Fuse multimodal results
            logger.info("Step 6: Fusing multimodal results...")
            final_result = fusion_engine.fuse_results(
                text_result, 
                video_result, 
                audio_result
            )
            
            # Prepare response
            response = {
                'success': True,
                'video_info': {
                    'title': download_info.get('title', 'Unknown'),
                    'duration': download_info.get('duration', 0),
                    'video_id': video_id
                },
                'final_emotion': {
                    'emotion': final_result['dominant_emotion'],
                    'confidence': final_result['confidence']
                },
                'modality_results': {
                    'text': {
                        'emotion': text_result['dominant_emotion'],
                        'confidence': text_result.get('confidence_scores', {}).get(text_result['dominant_emotion'], 0),
                        'reasoning': text_result.get('reasoning', '')
                    },
                    'video': {
                        'emotion': video_result['dominant_emotion'],
                        'frames_processed': video_result.get('total_frames_processed', 0),
                        'distribution': video_result.get('emotion_distribution', {})
                    },
                    'audio': {
                        'emotion': audio_result['dominant_emotion'],
                        'confidence': audio_result.get('confidence_scores', {}).get(audio_result['dominant_emotion'], 0)
                    }
                },
                'all_emotion_scores': final_result.get('emotion_scores', {})
            }
            
            logger.info(f"✅ Analysis complete. Final emotion: {final_result['dominant_emotion']}")
            return jsonify(response), 200
            
        finally:
            # Clean up downloaded files
            if video_id:
                try:
                    logger.info("Cleaning up temporary files...")
                    youtube_processor.cleanup(video_id)
                except Exception as cleanup_error:
                    logger.warning(f"Error during cleanup: {str(cleanup_error)}")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        return jsonify({
            'error': 'Model files not found. Please ensure all model files are in the correct directories.',
            'details': str(e)
        }), 500
        
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return jsonify({
            'error': 'Configuration error',
            'details': str(e)
        }), 500
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error processing video: {str(e)}")
        logger.error(error_details)
        return jsonify({
            'error': 'Failed to process YouTube video',
            'details': str(e),
            'traceback': error_details if app.debug else None
        }), 500

@app.route('/analyze-text', methods=['POST'])
def analyze_text_only():
    """Analyze text sentiment only (for testing)"""
    try:
        initialize_processors()
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text in request body'}), 400
        
        result = text_analyzer.analyze_sentiment(data['text'])
        return jsonify({
            'success': True,
            'result': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({
            'error': 'Failed to analyze text',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

