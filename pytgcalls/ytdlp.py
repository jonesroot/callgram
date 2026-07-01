import asyncio
import logging
import re
import shlex
import shutil
from typing import Optional, Tuple, Dict, List

from .exceptions import YtDlpError
from .ffmpeg import cleanup_commands
from .list_to_cmd import list_to_cmd
from .types.raw import VideoParameters

py_logger = logging.getLogger('pytgcalls')


class YtDlpCache:
    """
    A simple cache to store URL extraction results in order to avoid...
    repeated calls to yt-dlp that could trigger rate limits.
    """
    def __init__(self, ttl: float = 300.0):
        self._cache: Dict[str, Tuple[float, Tuple[str, str]]] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[Tuple[str, str]]:
        if key in self._cache:
            timestamp, value = self._cache[key]
            if asyncio.get_event_loop().time() - timestamp < self._ttl:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Tuple[str, str]) -> None:
        self._cache[key] = (asyncio.get_event_loop().time(), value)

    def clear(self) -> None:
        self._cache.clear()


# Initialize global cache for YtDlp instance
_extractor_cache = YtDlpCache(ttl=300.0)


async def _run_command_async(commands: List[str], timeout: float = 20.0) -> Tuple[int, str, str]:
    """
    Run system commands asynchronously using the native asyncio subprocess.
    Reduce thread-switching overhead and handle timeouts safely.
    """
    if not shutil.which(commands[0]):
        raise YtDlpError(f"Executable '{commands[0]}' was not found on the system.")

    try:
        proc = await asyncio.create_subprocess_exec(
            commands[0],
            *commands[1:],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), 
                timeout=timeout
            )
            return proc.returncode or 0, stdout_bytes.decode('utf-8', errors='ignore'), stderr_bytes.decode('utf-8', errors='ignore')
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise TimeoutError("The yt-dlp process exceeded the specified time limit.")
            
    except Exception as e:
        if not isinstance(e, TimeoutError):
            raise YtDlpError(f"Failed to execute command: {str(e)}") from e
        raise


class YtDlp:
    """
    A modern wrapper for yt-dlp that is efficient and resistant to blocking.
    """
    # Regulation of expression covering YouTube Music, Shorts, mobile, etc.
    YOUTUBE_REGX = re.compile(
        r'^(?:https?://)?(?:www\.|m\.|music\.)?'
        r'(?:youtu\.be/|youtube(?:-nocookie)?\.com/'
        r'(?:embed/|v/|watch\?v=|watch\?.+&v=|shorts/|live/))'
        r'([\w-]{11})',
        re.IGNORECASE
    )

    @staticmethod
    def is_valid(link: str) -> bool:
        """
        Validate whether the link is a valid YouTube URL.
        """
        if not link:
            return False
        return bool(YtDlp.YOUTUBE_REGX.match(link))

    @staticmethod
    async def extract(
        link: Optional[str],
        video_parameters: VideoParameters,
        add_commands: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracting audio and video URLs from YouTube links using yt-dlp.
        """
        if not link:
            return None, None

        if not YtDlp.is_valid(link):
            raise YtDlpError("Invalid YouTube link format")

        cache_key = f"{link}_{video_parameters.width}x{video_parameters.height}_{add_commands or ''}"
        if use_cache:
            cached_res = _extractor_cache.get(cache_key)
            if cached_res:
                py_logger.debug(f"Using the extracted results from the cache to: {link}")
                return cached_res

        max_res = min(video_parameters.width, video_parameters.height)
        commands = [
            'yt-dlp',
            '-g',
            '-f', f'bestvideo[height<={max_res}][vcodec~="(vp09|avc1)"]+bestaudio/best',
            '--no-warnings',
        ]

        if add_commands:
            cleaned = await cleanup_commands(
                shlex.split(add_commands),
                'yt-dlp',
                ['-f', '-g', '--no-warnings'],
            )
            commands.extend(cleaned)

        commands.append(link)

        py_logger.debug(f'Executing command: "{list_to_cmd(commands)}"')

        try:
            returncode, stdout, stderr = await _run_command_async(commands, timeout=25.0)

            if returncode != 0:
                error_msg = stderr.strip() if stderr else 'Unknown error'
                raise YtDlpError(f"yt-dlp error: {error_msg}")

            data = [line.strip() for line in stdout.strip().split('\n') if line.strip()]

            if not data:
                raise YtDlpError('Stream URL not found in yt-dlp output.')

            # yt-dlp returns the video on the first line and the audio on the second line (if separate).
            video_url = data[0]
            audio_url = data[1] if len(data) >= 2 else data[0]

            result = (video_url, audio_url)
            if use_cache:
                _extractor_cache.set(cache_key, result)

            return result

        except TimeoutError as e:
            raise YtDlpError("yt-dlp search timed out") from e


class YtDlpBatch:
    """
    YouTube batch extractor with concurrency management (semaphore).
    """

    def __init__(self, max_concurrent: int = 3):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_one(
        self,
        link: str,
        video_parameters: VideoParameters,
        add_commands: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[Exception]]:
        """
        Extracting a single link under Semaphore supervision.
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
        links: List[str],
        video_parameters: VideoParameters,
        add_commands: Optional[str] = None,
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Exception]]]:
        """
        Safely extract multiple links concurrently.
        """
        tasks = [
            self.extract_one(link, video_parameters, add_commands)
            for link in links
        ]
        return await asyncio.gather(*tasks)


async def extract_with_retry(
    link: str,
    video_parameters: VideoParameters,
    add_commands: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Tuple[Optional[str], Optional[str]]:
    """
    URL extraction with automatic recovery/retry mechanism.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return await YtDlp.extract(link, video_parameters, add_commands)
        except YtDlpError as e:
            last_error = e
            if attempt < max_retries - 1:
                py_logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {retry_delay} seconds .."
                )
                await asyncio.sleep(retry_delay)
            else:
                py_logger.error(f"Failed to extract after {max_retries} attempts: {e}")

    raise last_error if last_error else YtDlpError("Failed to process the link")


async def extract_with_fallback_formats(
    link: str,
    video_parameters: VideoParameters,
    add_commands: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try extracting using various format combinations step-by-step if it fails.
    """
    format_options = [
        'bestvideo[vcodec~="(vp09|avc1)"]+bestaudio/best',
        'bestvideo+bestaudio/best',
        'best',
    ]

    for fmt in format_options:
        try:
            custom_commands = f'-f {fmt}'
            if add_commands:
                custom_commands += f' {add_commands}'
            return await YtDlp.extract(link, video_parameters, custom_commands, use_cache=False)
        except YtDlpError as e:
            py_logger.debug(f"Alternative format '{fmt}' failed: {e}")
            continue

    raise YtDlpError("All streaming format options failed to extract.")


def is_playlist(link: str) -> bool:
    """
    Checks whether the provided link detects a YouTube playlist.
    """
    playlist_patterns = [
        r'[?&]list=',
        r'/playlist\?',
    ]
    return any(re.search(pattern, link) for pattern in playlist_patterns)


async def extract_playlist_urls(
    playlist_link: str,
    max_videos: Optional[int] = None,
) -> List[str]:
    """
    Extracts a list of individual video URLs from a YouTube playlist.
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
        returncode, stdout, stderr = await _run_command_async(commands, timeout=35.0)

        if returncode != 0:
            raise YtDlpError(stderr.strip() if stderr else 'Failed to process playlist')

        video_ids = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
        
        video_urls = [
            f'https://www.youtube.com/watch?v={vid_id}'
            for vid_id in video_ids if vid_id
        ]

        return video_urls

    except TimeoutError as e:
        raise YtDlpError("Playlist extraction timed out.") from e
