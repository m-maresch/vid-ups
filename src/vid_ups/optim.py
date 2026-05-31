import io

import openvino as ov
import torch


def compile(model, dummy_input):
    print("Exporting model to memory buffer...")
    onnx_buffer = io.BytesIO()
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
    ov_core = ov.Core()
    ov_model = ov_core.read_model(onnx_buffer.read(), b"")
    compiled_model = ov_core.compile_model(
        ov_model,
        "CPU",
        config={
            "INFERENCE_PRECISION_HINT": "FP32",
            "INFERENCE_NUM_THREADS": 8,
            "PERFORMANCE_HINT": "LATENCY",  # seems to work better than "THROUGHPUT"
        },
    )

    return compiled_model
