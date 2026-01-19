import os
import yt_dlp
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YouTubeProcessor:
    def __init__(self, temp_dir="temp_youtube"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def download_video(self, youtube_url):
        """
        Download video and extract audio from YouTube URL
        Returns paths to video and audio files
        """
        try:
            video_id = self._extract_video_id(youtube_url)
            video_path = os.path.join(self.temp_dir, f"{video_id}_video.mp4")
            audio_path = os.path.join(self.temp_dir, f"{video_id}_audio.wav")
            
            # Configure yt-dlp options
            ydl_opts_video = {
                'format': 'best[height<=720]',  # Download best quality up to 720p
                'outtmpl': video_path.replace('.mp4', '.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            ydl_opts_audio = {
                'format': 'bestaudio/best',
                'outtmpl': audio_path.replace('.wav', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            # Download video
            logger.info(f"Downloading video from: {youtube_url}")
            actual_video_path = None
            with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
                ydl.download([youtube_url])
                # Get the actual downloaded file path
                info = ydl.extract_info(youtube_url, download=False)
                downloaded_ext = info.get('ext', 'mp4')
                actual_video_path = video_path.replace('.mp4', f'.{downloaded_ext}')
                if not os.path.exists(actual_video_path):
                    # Try to find the file with the video_id
                    import glob
                    pattern = os.path.join(self.temp_dir, f"{video_id}_video.*")
                    matches = glob.glob(pattern)
                    if matches:
                        actual_video_path = matches[0]
            
            if actual_video_path and os.path.exists(actual_video_path):
                video_path = actual_video_path
            
            # Download audio
            logger.info("Extracting audio...")
            actual_audio_path = None
            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([youtube_url])
                # Audio should be converted to wav by postprocessor
                actual_audio_path = audio_path
                if not os.path.exists(actual_audio_path):
                    # Try to find the wav file
                    import glob
                    pattern = os.path.join(self.temp_dir, f"{video_id}_audio.*")
                    matches = glob.glob(pattern)
                    if matches:
                        actual_audio_path = matches[0]
            
            if actual_audio_path and os.path.exists(actual_audio_path):
                audio_path = actual_audio_path
            
            # Get video metadata
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                title = info.get('title', 'Unknown')
                description = info.get('description', '')
                duration = info.get('duration', 0)
            
            logger.info(f"✅ Video downloaded: {title}")
            
            return {
                'video_path': video_path,
                'audio_path': audio_path,
                'title': title,
                'description': description,
                'duration': duration,
                'video_id': video_id
            }
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise
    
    def get_transcript(self, youtube_url):
        """
        Extract transcript/subtitles from YouTube video
        """
        try:
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'subtitlesformat': 'vtt',
                'skip_download': True,
                'quiet': True,
            }
            
            transcript_text = ""
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # Try to get manual subtitles first, then auto-generated
                if 'subtitles' in info and 'en' in info['subtitles']:
                    subtitle_url = info['subtitles']['en'][0]['url']
                elif 'automatic_captions' in info and 'en' in info['automatic_captions']:
                    subtitle_url = info['automatic_captions']['en'][0]['url']
                else:
                    logger.warning("No subtitles available, will use description")
                    return info.get('description', '')
                
                # Download and parse VTT file
                import urllib.request
                import re
                
                vtt_content = urllib.request.urlopen(subtitle_url).read().decode('utf-8')
                
                # Remove VTT formatting and extract text
                lines = vtt_content.split('\n')
                for line in lines:
                    # Skip timestamps and VTT metadata
                    if '-->' in line or line.strip().startswith('<') or not line.strip():
                        continue
                    # Remove HTML tags
                    line = re.sub(r'<[^>]+>', '', line)
                    if line.strip():
                        transcript_text += line.strip() + " "
            
            logger.info(f"✅ Transcript extracted ({len(transcript_text)} characters)")
            return transcript_text.strip()
            
        except Exception as e:
            logger.warning(f"Could not extract transcript: {str(e)}")
            # Fallback to description
            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    return info.get('description', '')
            except:
                return ""
    
    def _extract_video_id(self, url):
        """Extract video ID from YouTube URL (supports regular videos and Shorts)"""
        import re
        patterns = [
            r'(?:youtube\.com\/shorts\/)([0-9A-Za-z_-]{11})',  # YouTube Shorts
            r'(?:youtube\.com\/watch\?v=)([0-9A-Za-z_-]{11})',  # Regular watch URL
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',  # Short URL
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Generic pattern
            r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed URL
            r'(?:v\/)([0-9A-Za-z_-]{11})',  # Alternative format
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return "unknown"
    
    def cleanup(self, video_id):
        """Clean up temporary files"""
        try:
            import glob
            pattern = os.path.join(self.temp_dir, f"{video_id}_*")
            for file in glob.glob(pattern):
                os.remove(file)
                logger.info(f"Cleaned up: {file}")
        except Exception as e:
            logger.warning(f"Error cleaning up files: {str(e)}")

