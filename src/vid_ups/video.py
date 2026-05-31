import subprocess


def start_reader(input_video):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            input_video,
            "-f",
            "image2pipe",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def start_writer(output_video, input_video, width, height):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-i",
            input_video,
            "-map",
            "0:v:0",  # Take video from the 1st input (stdin)
            "-map",
            "1:a:0",  # Take audio from the 2nd input (input_video)
            "-c:v",
            "h264_videotoolbox",
            "-c:a",
            "copy",
            output_video,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
