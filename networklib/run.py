from make_model import MakeModel
from inference_loading import convert_pytorch_to_onnx


if __name__ == "__main__":

    P = MakeModel(
        filename='../assets/data.csv',
        stock_ids=['NVDA'],
        train_end_date='2023-12-01',
        test_start_date='2024-02-01',
        start_date='2012-01-01',
        batch_size=128,
        epochs=10,
        lr=0.001,
        weight_decay=0.0001
    )

    MODEL = P.train()
    convert_pytorch_to_onnx(MODEL, onnx_file_path='../assets/model_3.onnx')
