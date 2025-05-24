from yt_dlp import YoutubeDL
import os

class VideoSourceManager:
    def __init__(self, source):
        self.source = source
        self.is_youtube = self._is_youtube_link(source)
        self.is_webcam = isinstance(source, int)

    def _is_youtube_link(self, url: str) -> bool:
        return isinstance(url, str) and (url.startswith("http://") or url.startswith("https://"))

    def get_streamable_path(self):
        if self.is_webcam:
            return self.source  # OpenCV can handle webcam index directly
        elif self.is_youtube:
            return self._get_youtube_stream_url()
        elif os.path.exists(self.source):
            return self.source
        else:
            raise ValueError("Unsupported or invalid video source.")

    def _get_youtube_stream_url(self):
        with YoutubeDL({'format': 'best'}) as ydl:
            info = ydl.extract_info(self.source, download=False)
            return info['url']
