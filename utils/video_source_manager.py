from yt_dlp import YoutubeDL
import os

class VideoSourceManager:
    def __init__(self, source: str):
        self.source = source
        self.is_youtube = self._is_youtube_link(source)

    def _is_youtube_link(self, url: str) -> bool:
        return url.startswith("http://") or url.startswith("https://")

    def get_streamable_path(self) -> str:
        if self.is_youtube:
            return self._get_youtube_stream_url()
        elif os.path.exists(self.source):
            return self.source
        else:
            raise ValueError("Invalid video source.")

    def _get_youtube_stream_url(self) -> str:
        with YoutubeDL({'format': 'best'}) as ydl:
            info = ydl.extract_info(self.source, download=False)
            return info['url']
