import asyncio
import logging
import re
import shlex
import subprocess
from typing import Optional
from typing import Tuple

from .exceptions import YtDlpError
from .ffmpeg import cleanup_commands
from .list_to_cmd import list_to_cmd
from .types.raw import VideoParameters

py_logger = logging.getLogger('pytgcalls')


class YtDlp:
    """
    Modern YouTube-DL wrapper using asyncio.to_thread.

    Compatible with Python 3.11+ and uvloop.
    """

    YOUTUBE_REGX = re.compile(
        r'^((?:https?:)?//)?((?:www|m)\.)?'
        r'(youtube(-nocookie)?\.com|youtu.be)'
        r'(/(?:[\w\-]+\?v=|embed/|live/|v/)?)'
        r'([\w\-]+)(\S+)?$',
    )

    @staticmethod
    def is_valid(link: str) -> bool:
        """
        Check if the provided link is a valid YouTube URL.

        Args:
            link: URL string to validate

        Returns:
            True if valid YouTube URL, False otherwise
        """
        return bool(YtDlp.YOUTUBE_REGX.match(link))

    @staticmethod
    async def extract(
        link: Optional[str],
        video_parameters: VideoParameters,
        add_commands: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract video and audio URLs from YouTube link using yt-dlp.

        Modern implementation using asyncio.to_thread instead of create_subprocess_exec.

        Args:
            link: YouTube video URL
            video_parameters: Video parameters for quality selection
            add_commands: Additional yt-dlp commands (optional)

        Returns:
            Tuple of (video_url, audio_url) or (None, None) if link is None

        Raises:
            YtDlpError: If yt-dlp fails or is not installed
            TimeoutError: If extraction takes too long
        """
        if link is None:
            return None, None

        commands = [
            'yt-dlp',
            '-g',
            '-f',
            'bestvideo[vcodec~="(vp09|avc1)"]+m4a/best',
            '-S',
            f'res:{min(video_parameters.width, video_parameters.height)}',
            '--no-warnings',
        ]

        if add_commands:
            commands += await cleanup_commands(
                shlex.split(add_commands),
                'yt-dlp',
                ['-f', '-g', '--no-warnings'],
            )

        commands.append(link)

        py_logger.log(
            logging.DEBUG,
            f'Running with "{list_to_cmd(commands)}" command',
        )

        try:
            async with asyncio.timeout(20):
                result = await asyncio.to_thread(
                    _run_ytdlp,
                    commands,
                )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else 'Unknown error'
                raise YtDlpError(error_msg)

            data = result.stdout.strip().split('\n')

            if not data or not data[0]:
                raise YtDlpError('No video URLs found')

            video_url = data[0]
            audio_url = data[1] if len(data) >= 2 else data[0]

            return video_url, audio_url

        except FileNotFoundError as e:
            raise YtDlpError('yt-dlp is not installed on your system') from e
        except subprocess.TimeoutExpired:
            raise YtDlpError('yt-dlp process timeout')
        except TimeoutError:
            raise YtDlpError('yt-dlp process timeout (asyncio)')


def _run_ytdlp(commands: list) -> subprocess.CompletedProcess:
    """
    Synchronous wrapper for yt-dlp execution.

    Designed to be used with asyncio.to_thread for non-blocking operation.

    Args:
        commands: List of command arguments

    Returns:
        CompletedProcess instance with stdout and stderr

    Raises:
        FileNotFoundError: If yt-dlp is not installed
        subprocess.TimeoutExpired: If command times out
    """
    return subprocess.run(
        commands,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


class YtDlpBatch:
    """
    Batch YouTube URL extractor with concurrency control.

    Useful for extracting multiple videos simultaneously with rate limiting.
    """

    def __init__(self, max_concurrent: int = 3):
        """
        Initialize batch extractor.

        Args:
            max_concurrent: Maximum number of concurrent extractions
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_one(
        self,
        link: str,
        video_parameters: VideoParameters,
        add_commands: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[Exception]]:
        """
        Extract single video with semaphore control.

        Args:
            link: YouTube video URL
            video_parameters: Video parameters
            add_commands: Additional yt-dlp commands

        Returns:
            Tuple of (video_url, audio_url, error)
        """
        async with self.semaphore:
            try:
                video_url, audio_url = await YtDlp.extract(
                    link,
                    video_parameters,
                    add_commands,
                )
                return video_url, audio_url, None
            except Exception as e:
                return None, None, e

    async def extract_many(
        self,
        links: list[str],
        video_parameters: VideoParameters,
        add_commands: Optional[str] = None,
    ) -> list[Tuple[Optional[str], Optional[str], Optional[Exception]]]:
        """
        Extract multiple videos concurrently.

        Args:
            links: List of YouTube video URLs
            video_parameters: Video parameters
            add_commands: Additional yt-dlp commands

        Returns:
            List of (video_url, audio_url, error) tuples
        """
        tasks = [
            self.extract_one(link, video_parameters, add_commands)
            for link in links
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results


async def extract_with_retry(
    link: str,
    video_parameters: VideoParameters,
    add_commands: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract YouTube URLs with automatic retry on failure.

    Args:
        link: YouTube video URL
        video_parameters: Video parameters
        add_commands: Additional yt-dlp commands
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Tuple of (video_url, audio_url)

    Raises:
        YtDlpError: If all retry attempts fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return await YtDlp.extract(link, video_parameters, add_commands)
        except YtDlpError as e:
            last_error = e
            if attempt < max_retries - 1:
                py_logger.warning(
                    f"yt-dlp attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
            else:
                py_logger.error(
                    f"yt-dlp failed after {max_retries} attempts: {e}"
                )

    raise last_error if last_error else YtDlpError("Unknown error")


async def extract_with_fallback_formats(
    link: str,
    video_parameters: VideoParameters,
    add_commands: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract YouTube URLs with fallback to different format options.

    Tries multiple format strings if the first one fails.

    Args:
        link: YouTube video URL
        video_parameters: Video parameters
        add_commands: Additional yt-dlp commands

    Returns:
        Tuple of (video_url, audio_url)

    Raises:
        YtDlpError: If all format options fail
    """
    format_options = [
        'bestvideo[vcodec~="(vp09|avc1)"]+m4a/best',
        'bestvideo+bestaudio/best',
        'best',
    ]

    for fmt in format_options:
        try:
            custom_commands = f'-f {fmt}'
            if add_commands:
                custom_commands += f' {add_commands}'
            return await YtDlp.extract(link, video_parameters, custom_commands)

        except YtDlpError as e:
            py_logger.debug(f"Format '{fmt}' failed: {e}")
            continue

    raise YtDlpError("All format options failed")


def is_playlist(link: str) -> bool:
    """
    Check if the link is a YouTube playlist.

    Args:
        link: URL to check

    Returns:
        True if link is a playlist
    """
    playlist_patterns = [
        r'[?&]list=',
        r'/playlist\?',
    ]
    return any(re.search(pattern, link) for pattern in playlist_patterns)


async def extract_playlist_urls(
    playlist_link: str,
    max_videos: Optional[int] = None,
) -> list[str]:
    """
    Extract individual video URLs from a YouTube playlist.

    Args:
        playlist_link: YouTube playlist URL
        max_videos: Maximum number of videos to extract (None for all)

    Returns:
        List of video URLs

    Raises:
        YtDlpError: If extraction fails
    """
    commands = [
        'yt-dlp',
        '--flat-playlist',
        '--get-id',
        '--no-warnings',
    ]

    if max_videos:
        commands.extend(['--playlist-end', str(max_videos)])

    commands.append(playlist_link)

    try:
        async with asyncio.timeout(30):
            result = await asyncio.to_thread(
                subprocess.run,
                commands,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        if result.returncode != 0:
            raise YtDlpError(result.stderr or 'Failed to extract playlist')

        video_ids = result.stdout.strip().split('\n')
        video_urls = [
            f'https://www.youtube.com/watch?v={vid_id}'
            for vid_id in video_ids if vid_id
        ]

        return video_urls

    except FileNotFoundError as e:
        raise YtDlpError('yt-dlp is not installed on your system') from e
    except (subprocess.TimeoutExpired, TimeoutError):
        raise YtDlpError('yt-dlp playlist extraction timeout')
