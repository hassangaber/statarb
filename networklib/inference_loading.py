import torch


def convert_pytorch_to_onnx(
    model: torch.nn.Module, input_size: tuple[int] = (64, 54), onnx_file_path: str = "../assets/model.onnx"
) -> None:
    model.eval()
    example_input = torch.randn(input_size)
    torch.onnx.export(
        model,
        example_input,
        onnx_file_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model has been converted to ONNX and saved to {onnx_file_path}")
