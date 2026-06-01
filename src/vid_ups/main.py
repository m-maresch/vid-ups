import argparse
import os
import sys
import time

import cv2
import numpy as np
import openvino as ov
import torch

from model import load_model
from optim import compile
from video import start_reader, start_writer

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False

frame_count = 0
start_time = time.perf_counter()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Video Upscaler: Run RealESRGAN or Spandrel models."
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to the input video file",
    )

    # Create subparsers for the two modes
    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Choose the mode"
    )

    # Mode 1: RealESRGAN
    parser_realesrgan = subparsers.add_parser(
        "realesrgan", help="Load RealESRGAN, use a specific layer retention"
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
        "spandrel", help="Load a model using Spandrel"
    )
    parser_spandrel.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the Spandrel-supported .pth file",
    )

    return parser.parse_args()


def write_upscaled_frame(request, writer):
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


def process_video(reader, infer_queue, writer, width, height):
    infer_queue.set_callback(lambda request, _: write_upscaled_frame(request, writer))

    in_frame_size = width * height * 3
    print("Processing video stream...")
    try:
        while True:
            in_bytes = reader.stdout.read(in_frame_size)
            if not in_bytes or len(in_bytes) != in_frame_size:
                break

            img_np = np.frombuffer(in_bytes, dtype=np.uint8).reshape((height, width, 3))
            img_tensor = img_np.astype(np.float32).transpose(2, 0, 1) / 255.0
            img_tensor = np.expand_dims(img_tensor, axis=0)

            infer_queue.start_async({0: img_tensor})
        infer_queue.wait_all()
    finally:
        reader.terminate()
        if writer.stdin:
            writer.stdin.close()
        writer.wait()


def main():
    args = parse_arguments()

    print(f"Target Video: {args.video}")
    if args.mode == "realesrgan":
        print("Mode: RealESRGAN")
        print(f"Slicing architecture to keep {args.keep_layers} layers.")
    elif args.mode == "spandrel":
        print("Mode: Spandrel")
    print(f"Loading model: {args.model}")

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = load_model(args)

    dummy_input = torch.randn(1, 3, height, width)
    compiled_model = compile(model, dummy_input)
    infer_queue = ov.AsyncInferQueue(compiled_model)

    reader = start_reader(input_video)
    out_width, out_height = width * 2, height * 2
    writer = start_writer(output_video, input_video, out_width, out_height)

    process_video(reader, infer_queue, writer, width, height)


if __name__ == "__main__":
    main()
