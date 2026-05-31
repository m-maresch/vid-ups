import argparse
import io
import os
import subprocess
import sys
import time

import cv2
import openvino as ov
import torch
import numpy as np

from basicsr.archs.rrdbnet_arch import RRDBNet
from spandrel import ModelLoader

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Video Upscaler: Run RealESRGAN or Spandrel models."
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to the input video file",
    )

    # Create subparsers for the two distinct modes
    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Choose the processing mode"
    )

    # Mode 1: RealESRGAN
    parser_realesrgan = subparsers.add_parser(
        "realesrgan", help="Use RealESRGAN with specific layer retention"
    )
    parser_realesrgan.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the RealESRGAN .pth file",
    )
    parser_realesrgan.add_argument(
        "-k",
        "--keep-layers",
        type=int,
        required=True,
        help="Number of model layers to keep",
    )

    # Mode 2: Spandrel
    parser_spandrel = subparsers.add_parser(
        "spandrel", help="Load a Spandrel-based model"
    )
    parser_spandrel.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the Spandrel-supported .pth file",
    )

    return parser.parse_args()


args = parse_arguments()

print(f"Target Video: {args.video}\n")
print(f"Loading model: {args.model}")

if args.mode == "realesrgan":
    print("Mode: RealESRGAN")
    print(f"Slicing architecture to keep {args.keep_layers} layers.")
elif args.mode == "spandrel":
    print("Mode: Spandrel")

input_video = args.video
model_pth = args.model

if not os.path.exists(input_video):
    print(f"Error: The file '{input_video}' does not exist.")
    sys.exit(1)

if not os.path.exists(model_pth):
    print(f"Error: The file '{model_pth}' does not exist.")
    sys.exit(1)

name, ext = os.path.splitext(input_video)
output_video = f"{name}_2x{ext}"

cap = cv2.VideoCapture(input_video)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

OUT_WIDTH, OUT_HEIGHT = WIDTH * 2, HEIGHT * 2
FRAME_SIZE = WIDTH * HEIGHT * 3

if args.mode == "realesrgan":
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    checkpoint = torch.load(model_pth, map_location="cpu")
    model.load_state_dict(checkpoint.get("params_ema", checkpoint), strict=True)
    model.body = torch.nn.Sequential(*[model.body[i] for i in range(args.keep_layers)])
elif args.mode == "spandrel":
    model = ModelLoader().load_from_file(model_pth).model

model.eval()

print("Exporting model to virtual memory buffer...")
onnx_buffer = io.BytesIO()
dummy_input = torch.randn(1, 3, HEIGHT, WIDTH)

torch.onnx.export(
    model,
    dummy_input,
    onnx_buffer,
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
)
onnx_buffer.seek(0)  # Reset buffer pointer to the beginning

print("Converting and compiling onto Intel...")
core = ov.Core()
# Read the model directly from our virtual binary buffer
ov_model = core.read_model(onnx_buffer.read(), b"")
compiled_model = core.compile_model(
    ov_model,
    "CPU",
    config={
        "INFERENCE_PRECISION_HINT": "FP32",
        "INFERENCE_NUM_THREADS": 8,
        "PERFORMANCE_HINT": "LATENCY",  # seems to work better than "THROUGHPUT"
    },
)
infer_queue = ov.AsyncInferQueue(compiled_model)

reader = subprocess.Popen(
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
writer = subprocess.Popen(
    [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{OUT_WIDTH}x{OUT_HEIGHT}",
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

print("Processing video stream...")

frame_count = 0
start_time = time.perf_counter()


def completion_callback(request, user_data):
    global frame_count, start_time
    output_node = request.get_output_tensor(0)
    out_tensor = output_node.data

    out_np = np.squeeze(out_tensor, axis=0).transpose(1, 2, 0)
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)

    writer.stdin.write(out_np.tobytes())
    frame_count += 1

    elapsed = time.perf_counter() - start_time
    start_time = time.perf_counter()
    print(f"Frame {frame_count} processed in {elapsed:.2f}s ({1 / elapsed:.2f} FPS)")


infer_queue.set_callback(completion_callback)

try:
    while True:
        # Read raw frame bytes from the input pipe
        in_bytes = reader.stdout.read(FRAME_SIZE)
        if not in_bytes or len(in_bytes) != FRAME_SIZE:
            break

        img_np = np.frombuffer(in_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        img_tensor = img_np.astype(np.float32).transpose(2, 0, 1) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)

        infer_queue.start_async({0: img_tensor})
    infer_queue.wait_all()
finally:
    reader.terminate()
    if writer.stdin:
        writer.stdin.close()
    writer.wait()
