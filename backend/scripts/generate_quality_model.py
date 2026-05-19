"""Generate a minimal ONNX quality classifier model.

Produces a small model that takes [1,3,224,224] float32 input and outputs
[1,4] float32: [blur_score, glare_score, orientation_idx, brightness_idx].

This is a placeholder model using random weights. Replace with a real
trained MobileNetV3 export for production.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

def make_quality_model() -> onnx.ModelProto:
    # Input: [1, 3, 224, 224]
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    # Output: [1, 4]
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    # -- Global Average Pool to reduce spatial dims: [1,3,224,224] -> [1,3,1,1]
    gap = helper.make_node("GlobalAveragePool", ["input"], ["gap_out"], name="gap")

    # -- Reshape to [1, 3]
    shape_const = numpy_helper.from_array(
        np.array([1, 3], dtype=np.int64), name="shape_fc"
    )
    reshape = helper.make_node("Reshape", ["gap_out", "shape_fc"], ["flat"], name="reshape")

    # -- MatMul: [1,3] x [3,4] -> [1,4]
    np.random.seed(42)
    W_val = np.random.randn(3, 4).astype(np.float32) * 0.5
    W = numpy_helper.from_array(W_val, name="W")

    matmul = helper.make_node("MatMul", ["flat", "W"], ["mm_out"], name="matmul")

    # -- Bias: [1,4] + [4]
    B_val = np.array([0.6, 0.1, 0.0, 0.0], dtype=np.float32)
    B = numpy_helper.from_array(B_val, name="B")
    add = helper.make_node("Add", ["mm_out", "B"], ["add_out"], name="add")

    # -- Sigmoid on first two outputs would be ideal, but keep it simple
    # Just clamp with Sigmoid for the whole thing to keep values bounded
    sigmoid = helper.make_node("Sigmoid", ["add_out"], ["output"], name="sigmoid")

    graph = helper.make_graph(
        [gap, reshape, matmul, add, sigmoid],
        "quality_classifier",
        [X],
        [Y],
        initializer=[shape_const, W, B],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    import os
    import shutil

    model = make_quality_model()

    # Save to backend weights
    backend_path = os.path.join(os.path.dirname(__file__), "..", "weights", "quality_classifier.onnx")
    backend_path = os.path.abspath(backend_path)
    onnx.save(model, backend_path)
    print(f"Saved: {backend_path} ({os.path.getsize(backend_path)} bytes)")

    # Copy to frontend public/models
    frontend_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "id-capture", "public", "models", "quality_classifier.onnx")
    )
    os.makedirs(os.path.dirname(frontend_path), exist_ok=True)
    shutil.copy2(backend_path, frontend_path)
    print(f"Copied: {frontend_path} ({os.path.getsize(frontend_path)} bytes)")
