import asyncio
import logging
import os.path
import re
import shlex
import subprocess
from json import JSONDecodeError
from json import loads
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from functools import partial

from ntgcalls import FFmpegError

from .exceptions import ImageSourceFound
from .exceptions import InvalidVideoProportion
from .exceptions import LiveStreamFound
from .exceptions import NoAudioSourceFound
from .exceptions import NoVideoSourceFound
from .types.raw import AudioParameters
from .types.raw import VideoParameters


async def check_stream(
    ffmpeg_parameters: Optional[str],
    path: str,
    stream_parameters: Union[AudioParameters, VideoParameters],
    before_commands: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """
    Check stream properties using ffprobe.

    Modern implementation using asyncio.to_thread with subprocess.run.
    Compatible with Python 3.12+ and uvloop.
    """
    commands = await cleanup_commands(
        build_command(
            'ffprobe',
            ffmpeg_parameters,
            path,
            stream_parameters,
            before_commands,
            headers,
            False,
        ),
    )

    try:
        async with asyncio.timeout(20):
            result = await asyncio.to_thread(
                _run_subprocess,
                commands,
                capture_output=True,
                text=True,
                timeout=20,
            )
    except FileNotFoundError as e:
        raise FFmpegError('ffprobe not installed') from e
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"ffprobe timeout for: {path}")

    try:
        result_data = loads(result.stdout) if result.stdout else {}
        stream_list = result_data.get('streams', [])
        format_content = result_data.get('format', {})

        if 'No such file' in result.stderr:
            raise FileNotFoundError(f"Stream not found: {path}")

    except JSONDecodeError as e:
        raise FFmpegError(f"Invalid JSON output from ffprobe: {e}") from e

    have_video = False
    is_image = True
    have_audio = False
    have_valid_video = False

    original_width, original_height = 0, 0

    for stream in stream_list:
        codec_type = stream.get('codec_type', '')
        codec_name = stream.get('codec_name', '')
        image_codecs = {'png', 'jpeg', 'jpg', 'mjpeg'}

        if codec_type == 'video':
            is_image &= codec_name in image_codecs
            have_video = True
            original_width = int(stream.get('width', 0))
            original_height = int(stream.get('height', 0))
            if original_height and original_width:
                have_valid_video = True
        elif codec_type == 'audio':
            have_audio = True

    if isinstance(stream_parameters, VideoParameters):
        if not have_video:
            raise NoVideoSourceFound(path)
        if not have_valid_video:
            raise InvalidVideoProportion('Video proportion not found')

        ratio = float(original_width) / original_height
        new_w = min(original_width, stream_parameters.width)
        new_h = int(new_w / ratio)

        if (
            new_h > stream_parameters.height and
            stream_parameters.adjust_by_height
        ):
            new_h = stream_parameters.height
            new_w = int(new_h * ratio)

        new_w = new_w - 1 if new_w % 2 else new_w
        new_h = new_h - 1 if new_h % 2 else new_h

        stream_parameters.height = new_h
        stream_parameters.width = new_w

        if is_image:
            stream_parameters.frame_rate = 10
            raise ImageSourceFound(path)

    if isinstance(stream_parameters, AudioParameters) and not have_audio:
        raise NoAudioSourceFound(path)

    if 'duration' not in format_content:
        raise LiveStreamFound(path)


async def cleanup_commands(
    commands: List[str],
    process_name: Optional[str] = None,
    blacklist: Optional[List[str]] = None,
) -> List[str]:
    """
    Clean up FFmpeg commands by removing unsupported flags.

    Modern implementation using asyncio.to_thread.
    """
    if not commands:
        return []

    help_commands = [
        commands[0] if not process_name else process_name,
        '-h',
        'full',
    ]

    try:
        async with asyncio.timeout(20):
            result = await asyncio.to_thread(
                _run_subprocess,
                help_commands,
                capture_output=True,
                text=True,
                timeout=20,
            )

    except FileNotFoundError as e:
        raise FFmpegError(f'{commands[0]} not installed') from e
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Command timeout: {' '.join(help_commands)}")

    supported = re.findall(r'(?m)^ *(-\w+).*?\s+', result.stdout)
    supported.append('-i')
    supported_set = set(supported)

    new_commands = []
    ignore_next = False
    blacklist_set = set(blacklist) if blacklist else set()

    for v in commands:
        if not v:
            continue

        if v[0] == '-':
            ignore_next = (
                v not in supported_set or 
                v in blacklist_set
            )

        if not ignore_next:
            new_commands.append(v)
        elif v[0] != '-':
            ignore_next = False

    return new_commands


def _run_subprocess(
    commands: List[str],
    capture_output: bool = True,
    text: bool = True,
    timeout: Optional[float] = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Synchronous subprocess runner for use with asyncio.to_thread.

    This is a wrapper around subprocess.run that can be safely used
    with asyncio.to_thread in Python 3.12+.
    """
    return subprocess.run(
        commands,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        check=False,
        **kwargs,
    )


def build_command(
    name: str,
    ffmpeg_parameters: Optional[str],
    path: Optional[str],
    stream_parameters: Union[AudioParameters, VideoParameters],
    before_commands: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
    is_livestream: bool = False,
) -> List[str]:
    """
    Build FFmpeg/FFprobe command with proper parameters.
    """
    if not path:
        return []

    command = _get_stream_params(ffmpeg_parameters)

    if isinstance(stream_parameters, VideoParameters):
        command = command['video']
    else:
        command = command['audio']

    ffmpeg_command: List[str] = [name]

    ffmpeg_command.extend(command['start'])

    if (
        not os.path.exists(path) and 
        not is_livestream and 
        name == 'ffmpeg'
    ):
        ffmpeg_command.extend([
            '-reconnect', '1',
            '-reconnect_at_eof', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '2',
        ])

    if name == 'ffprobe':
        ffmpeg_command.extend([
            '-v', 'error',
            '-show_entries', 'stream=width,height,codec_type,codec_name',
            '-show_format',
            '-of', 'json',
        ])

    if before_commands:
        ffmpeg_command.extend(before_commands)

    if headers:
        for key, value in headers.items():
            ffmpeg_command.extend(['-headers', f'{key}: {value}'])

    ffmpeg_command.extend(['-i', f'{path}' if name == 'ffmpeg' else path])
    ffmpeg_command.extend(command['mid'])

    if name == 'ffmpeg':
        ffmpeg_command.extend(_build_ffmpeg_options(stream_parameters))
        ffmpeg_command.extend(command['end'])
        ffmpeg_command.append('pipe:1')
    else:
        ffmpeg_command.extend(command['end'])

    return ffmpeg_command


def _get_stream_params(command: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse stream parameters from custom FFmpeg parameter string.
    """
    arg_names = ['base', 'audio', 'video']
    command_args: Dict[str, List[str]] = {arg: [] for arg in arg_names}
    current_arg = arg_names[0]

    if command:
        for part in shlex.split(command):
            arg_name = part[2:] if part.startswith('--') else None
            if arg_name in arg_names:
                current_arg = arg_name
            else:
                command_args[current_arg].append(part)

    command_args_processed = {
        cmd: _extract_stream_params(command_args[cmd])
        for cmd in command_args
    }

    for arg in arg_names[1:]:
        for param_type in command_args_processed[arg_names[0]]:
            command_args_processed[arg][param_type].extend(
                command_args_processed[arg_names[0]][param_type]
            )

    del command_args_processed[arg_names[0]]

    return command_args_processed


def _extract_stream_params(command: List[str]) -> Dict[str, List[str]]:
    """
    Extract start, mid, and end parameters from command list.
    """
    arg_names = ['start', 'mid', 'end']
    command_args: Dict[str, List[str]] = {arg: [] for arg in arg_names}
    current_arg = arg_names[0]

    for part in command:
        arg_name = part[3:] if part.startswith('---') else None
        if arg_name in arg_names:
            current_arg = arg_name
        else:
            command_args[current_arg].append(part)

    return command_args


def _build_ffmpeg_options(
    stream_parameters: Union[AudioParameters, VideoParameters],
) -> List[str]:
    """
    Build FFmpeg output options based on stream parameters.
    """
    log_level = logging.getLogger('ffmpeg').level
    ffmpeg_level = 'info' if log_level == logging.DEBUG else 'quiet'

    options = ['-v', ffmpeg_level, '-f']

    if isinstance(stream_parameters, AudioParameters):
        options.extend([
            's16le',
            '-ac', str(stream_parameters.channels),
            '-ar', str(stream_parameters.bitrate),
        ])
    elif isinstance(stream_parameters, VideoParameters):
        options.extend([
            'rawvideo',
            '-r', str(stream_parameters.frame_rate),
            '-pix_fmt', 'yuv420p',
            '-vf', f'scale={stream_parameters.width}:{stream_parameters.height}',
        ])

    return options


def setup_uvloop() -> None:
    """
    Setup uvloop for better performance (optional).
    Call this at the start of your application.
    """
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        logging.warning("uvloop not installed, using default event loop")


async def check_stream_batch(
    streams: List[tuple],
    max_concurrent: int = 5,
) -> List[Union[Exception, None]]:
    """
    Check multiple streams concurrently with rate limiting.

    Args:
        streams: List of tuples (ffmpeg_parameters, path, stream_parameters, 
                 before_commands, headers)
        max_concurrent: Maximum number of concurrent checks

    Returns:
        List of results (None for success, Exception for failure)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_with_semaphore(args):
        async with semaphore:
            try:
                await check_stream(*args)
                return None
            except Exception as e:
                return e

    results = await asyncio.gather(
        *[check_with_semaphore(stream) for stream in streams],
        return_exceptions=False,
    )

    return results


class AsyncFFmpegProcess:
    """
    Modern async wrapper for FFmpeg process using asyncio.to_thread.

    This class provides non-blocking access to FFmpeg stdin/stdout
    without using create_subprocess_exec.
    """

    def __init__(self, commands: List[str]):
        self.commands = commands
        self.process: Optional[subprocess.Popen] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the FFmpeg process."""
        self.process = await asyncio.to_thread(
            subprocess.Popen,
            self.commands,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    async def write(self, data: bytes) -> None:
        """Write data to stdin asynchronously."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not started or stdin not available")

        await asyncio.to_thread(
            self.process.stdin.write,
            data,
        )
        await asyncio.to_thread(self.process.stdin.flush)

    async def read(self, size: int = -1) -> bytes:
        """Read data from stdout asynchronously."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Process not started or stdout not available")

        return await asyncio.to_thread(
            self.process.stdout.read,
            size,
        )

    async def read_stderr(self, size: int = -1) -> bytes:
        """Read data from stderr asynchronously."""
        if not self.process or not self.process.stderr:
            raise RuntimeError("Process not started or stderr not available")

        return await asyncio.to_thread(
            self.process.stderr.read,
            size,
        )

    async def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for process to complete."""
        if not self.process:
            raise RuntimeError("Process not started")

        try:
            if timeout:
                async with asyncio.timeout(timeout):
                    return await asyncio.to_thread(self.process.wait)
            else:
                return await asyncio.to_thread(self.process.wait)
        except TimeoutError:
            await self.terminate()
            raise

    async def terminate(self) -> None:
        """Terminate the process gracefully."""
        if not self.process:
            return

        await asyncio.to_thread(self.process.terminate)
        try:
            async with asyncio.timeout(5):
                await asyncio.to_thread(self.process.wait)
        except TimeoutError:
            await asyncio.to_thread(self.process.kill)
            await asyncio.to_thread(self.process.wait)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.terminate()
        return False
