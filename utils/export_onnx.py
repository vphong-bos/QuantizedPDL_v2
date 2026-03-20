import os
import onnxruntime as ort


def export_optimized_onnx_model(
    quant_weights,
    output_path,
    provider="CPUExecutionProvider",
    optimization_level="extended",   # "basic", "extended", "all"
):
    """
    Export an ONNX Runtime optimized model.

    Args:
        quant_weights (str): path to source .onnx
        output_path (str): path to save optimized model (.onnx or .ort)
        provider (str): ORT execution provider
        optimization_level (str): one of {"basic", "extended", "all"}

    Returns:
        str: saved path
    """
    if not quant_weights.lower().endswith(".onnx"):
        raise ValueError(f"Input must be .onnx, got: {quant_weights}")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    level_map = {
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if optimization_level not in level_map:
        raise ValueError(f"Invalid optimization_level: {optimization_level}")

    print(f"Exporting optimized ONNX model from: {quant_weights}")
    print(f"Saving to: {output_path}")
    print(f"Using provider: {provider}")
    print(f"Optimization level: {optimization_level}")

    so = ort.SessionOptions()
    so.graph_optimization_level = level_map[optimization_level]
    so.optimized_model_filepath = output_path

    try:
        _ = ort.InferenceSession(
            quant_weights,
            sess_options=so,
            providers=[provider],
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to export optimized model to '{output_path}'. "
            f"Try optimization_level='extended' instead of 'all', "
            f"or save as '.ort' format. Original error: {e}"
        ) from e

    print(f"Optimized model exported successfully: {output_path}")
    return output_path