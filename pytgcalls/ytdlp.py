import asyncio
import logging
import re
import shlex
import subprocess
from typing import Optional, Tuple

from .exceptions import YtDlpError
from .ffmpeg import cleanup_commands
from .list_to_cmd import list_to_cmd
from .types.raw import VideoParameters

py_logger = logging.getLogger("pytgcalls")


class YtDlp:
    YOUTUBE_REGX = re.compile(
        r'^((?:https?:)?//)?((?:www|m)\.)?'
        r'(youtube(-nocookie)?\.com|youtu.be)'
        r'(/(?:[\w\-]+\?v=|embed/|live/|v/)?)'
        r'([\w\-]+)(\S+)?$',
    )

    @staticmethod
    def is_valid(link: str) -> bool:
        return bool(YtDlp.YOUTUBE_REGX.match(link))

    @staticmethod
    async def extract(
        link: Optional[str],
        video_parameters: VideoParameters,
        add_commands: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if not link:
            return None, None

        commands = [
            "yt-dlp",
            "-g",
            "-f",
            'bestvideo[vcodec~="(vp09|avc1)"]+m4a/best',
            "-S",
            f"res:{min(video_parameters.width, video_parameters.height)}",
            "--no-warnings",
        ]

        if add_commands:
            commands += await cleanup_commands(
                shlex.split(add_commands),
                process_name="yt-dlp",
                blacklist=["-f", "-g", "--no-warnings"],
            )

        commands.append(link)
        command_str = list_to_cmd(commands)
        py_logger.debug(f'Running yt-dlp command: {command_str}')

        try:
            def run_ytdlp():
                return subprocess.run(
                    commands,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True,
                    timeout=20,
                )

            result = await asyncio.to_thread(run_ytdlp)

            if result.returncode != 0:
                raise YtDlpError(result.stderr.strip())

            output_lines = result.stdout.strip().split("\n")
            if not output_lines:
                raise YtDlpError("No video URLs found")

            return (
                output_lines[0],
                output_lines[1] if len(output_lines) > 1 else output_lines[0],
            )

        except subprocess.TimeoutExpired:
            raise YtDlpError("yt-dlp process timeout")
        except FileNotFoundError:
            raise YtDlpError("yt-dlp is not installed on your system")
