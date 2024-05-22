import onnxruntime as ort
import numpy as np


def run_inference_onnx(model_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Run inference on an ONNX model.

    Parameters:
        model_path (str): Path to the ONNX model.
        input_data (np.ndarray): Input data for the model.

    Returns:
        np.ndarray: Model predictions as probabilities.
    """
    sess = ort.InferenceSession(model_path)

    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=0)

    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: input_data.astype(np.float32)})
    probabilities = result[0]
    return probabilities
