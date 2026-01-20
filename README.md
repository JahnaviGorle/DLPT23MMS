# Multimodal Sentiment Analysis from YouTube Videos

An end-to-end backend system that analyzes emotions from YouTube videos using text (transcript), video (facial expressions), and audio features.

## Features

- **Text Analysis**: Extracts and analyzes video transcripts/descriptions using Google Gemini AI
- **Video Analysis**: Detects facial emotions from video frames using deep learning
- **Audio Analysis**: Analyzes audio features (MFCC) to detect emotions
- **Multimodal Fusion**: Combines results from all three modalities for accurate emotion detection

## Setup

### 1. Install Dependencies

```bash

py -3.11 -m venv venv
venv\Scripts\activate
#Verify Python Version

pip install -r requirements.txt
```

### 2. Install FFmpeg (Windows)

FFmpeg is required for video/audio processing. 

**❌ What you downloaded:** You downloaded the FFmpeg **source code** (`ffmpeg-8.0.1.tar.xz`), which needs to be compiled.

**✅ What you need:** Pre-built **Windows binaries** (already compiled)

**Correct Download Link:**
1. Go to: https://www.gyan.dev/ffmpeg/builds/
2. Download: **"ffmpeg-release-essentials.zip"** (or "ffmpeg-release-full.zip" for all features)
   - Direct link: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
3. Extract the ZIP file (you'll get a folder like `ffmpeg-6.x-essentials_build`)
4. Add FFmpeg to your Windows PATH:
   - Copy the `bin` folder path (e.g., `C:\ffmpeg-6.x-essentials_build\bin`)
   - Open Windows Settings → System → About → Advanced system settings
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and paste the bin folder path
   - Click OK on all windows
5. Verify installation: Open PowerShell/CMD and run:
   ```bash
   ffmpeg -version
   ```
   If it shows version info, FFmpeg is correctly installed!

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Model Files

Ensure you have the following model files in the correct directories:

- **Audio Model**: `MODELS/model_bi-lstm.keras`
- **Video Model**: `model/model.h5`

## Project Structure

```
MMS/
├── app.py                          # Main Flask backend API
├── youtube_processor.py            # YouTube video downloader and processor
├── audio_emotion_detector.py       # Audio emotion detection
├── emotion_detector.py             # Video (facial) emotion detection
├── text_sentiment_analyzer.py      # Text sentiment analysis using Gemini
├── multimodal_fusion.py            # Combines results from all modalities
├── gemini_chat.py                  # Gemini chat utilities
├── requirements.txt                # Python dependencies
└── .env                            # Environment variables (create this)
```

## API Usage

### Start the Server

```bash
python app.py
```

The server will run on `http://localhost:5000`

### Analyze YouTube Video

**Endpoint**: `POST /analyze`

**Request Body**:
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response**:
```json
{
  "success": true,
  "video_info": {
    "title": "Video Title",
    "duration": 120,
    "video_id": "VIDEO_ID"
  },
  "final_emotion": {
    "emotion": "happy",
    "confidence": 85.5
  },
  "modality_results": {
    "text": {
      "emotion": "happy",
      "confidence": 90.0,
      "reasoning": "..."
    },
    "video": {
      "emotion": "happy",
      "frames_processed": 15,
      "distribution": {...}
    },
    "audio": {
      "emotion": "happy",
      "confidence": 88.0
    }
  },
  "all_emotion_scores": {
    "happy": 85.5,
    "neutral": 10.2,
    ...
  }
}
```

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "message": "Multimodal Sentiment Analysis API is running"
}
```

### Analyze Text Only (Testing)

**Endpoint**: `POST /analyze-text`

**Request Body**:
```json
{
  "text": "This is a sample text to analyze"
}
```

## Example Usage with cURL

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

## Example Usage with Python

```python
import requests

url = "http://localhost:5000/analyze"
data = {
    "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Final Emotion: {result['final_emotion']['emotion']}")
print(f"Confidence: {result['final_emotion']['confidence']}%")
```

## Supported Emotions

The system detects the following emotions:
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise
- Neutral
- Calm

## Notes

- The system downloads videos temporarily and cleans them up after processing
- Video processing may take time depending on video length
- Ensure you have sufficient disk space for temporary video files
- The API uses weighted fusion (default: Text 30%, Video 40%, Audio 30%)

## Troubleshooting

1. **Model files not found**: Ensure model files are in the correct directories
2. **FFmpeg errors**: Install FFmpeg and ensure it's in your PATH
3. **Gemini API errors**: Check your API key in `.env` file
4. **Memory errors**: Process shorter videos or increase system RAM

