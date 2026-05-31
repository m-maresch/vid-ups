# vid-ups

`vid-ups` is a command-line tool for hardware-accelerated AI video upscaling on CPU targets (Intel).

By converting PyTorch weights (.pth) to an ONNX graph, compiling them for use with the optimized OpenVINO runtime, and orchestrating dual FFmpeg subprocess pipes, this script achieves faster execution and avoids the need to first convert the video into its individual frames on disk.

## Features

- **2 Modes**:
  - RealESRGAN Mode: Direct initialization of the RRDBNet architecture with customizable removal of layer blocks for performance tuning.
  - Spandrel Mode: Automatic model architecture detection and loading via Spandrel.
- **FFmpeg Piping**: Pipes raw video frames directly out of and back into FFmpeg. Retains the original audio track.
- **OpenVINO Optimization**: Compiles models directly to Intel-optimized execution units.
- **Asynchronous Inference**: Utilizes OpenVINO's `AsyncInferQueue` to process frames.

## Pipeline Architecture

The script processes video files through an execution pipeline:

    [Input Video] 
          │
          ▼ (FFmpeg Demuxer Pipe)
    [Raw Frames] 
          │
          ▼ (Normalization)
    [OpenVINO AsyncInferQueue] ◄─── (ONNX Model via PyTorch)
          │
          ▼ (Completion Callback: Scale)
    [Upscaled Frames] 
          │
          ▼ (FFmpeg Muxer Pipe + Audio Mapping)
    [Output Video (2x)]

## Prerequisites

### System Dependencies
- **FFmpeg**: Must be installed and available in your system's PATH.

### Installation

Clone the repository:

```
git clone https://github.com/m-maresch/vid-ups
cd vid-ups
```

Create Python venv:

```
python -m venv .
source bin/activate
```

Then install the dependencies into the venv as per the `requirements.txt`.

## Usage

The script exposes a command-line interface with explicit subcommands for each mode. The output file is automatically saved at 2x scale with the suffix _2x.

    ./vid_ups.sh <path_to_video> [mode] [options]

### Mode 1: RealESRGAN
For RRDBNet architectures where you want to remove layer blocks from the model to speed up processing.

    ./vid_ups.sh input.mp4 realesrgan --model weights/realesrgan.pth --keep-layers 10

**Arguments:**
- -m, --model (Required): Path to the RealESRGAN .pth checkpoint (see [here](https://github.com/xinntao/Real-ESRGAN/releases), e.g. `RealESRGAN_x2plus.pth`).
- -k, --keep-layers (Required): Integer count specifying how many blocks of the structural model body to retain.

### Mode 2: Spandrel
For loading checkpoints natively supported by Spandrel's model loader.

    ./vid_ups.sh input.mp4 spandrel --model weights/your_upscaler.pth

**Arguments:**
- -m, --model (Required): Path to the Spandrel-supported .pth file (see [here](https://github.com/chaiNNer-org/spandrel#single-image-super-resolution), e.g. `spanx2_ch52.pth`).

## Intel Performance Tuning

The script defaults to 8 execution threads. Experiment with setting `"PERFORMANCE_HINT": "THROUGHPUT"` or altering `INFERENCE_NUM_THREADS` to match your CPU's physical core count.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details. Third-party library notices are documented in `THIRD-PARTY-NOTICES.txt`.

## Acknowledgments

This project was developed with the assistance of Google Gemini 3.

## Dependencies

Thanks to everyone contributing to any of the following projects:

- OpenCV
- NumPy
- PyTorch
- BasicSR
- Spandrel and the various models
- RealESRGAN
- OpenVINO
- FFmpeg

