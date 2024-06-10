import pandas as pd
import torch
from information_features import EntropyFeatures
from is_series_stationary import process_series
from make_indicators import TechnicalIndicators
from misc import MiscFeatures
from MLP.dataset import TimeSeriesDataset
from MLP.make_target import TargetGenerator
from MLP.model import FeedForwardNN, LSTMModel
from preprocessing import filter_df_by_date, format_floats, scale_datasets
from torch.optim import SGD, AdamW, Adam
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

DATA = pd.read_csv('../data/bloomberg20240504.csv')

def CREATE_FEATURES(df: pd.DataFrame, save:bool=True) -> pd.DataFrame:

    print('Making Entropy Features...')
    EF=EntropyFeatures(data=DATA)
    df=EF.fit_transform()

    # print('Checking for Stationary Time-Series...')
    # df_staionary = process_series(df=df, alpha=0.05) # 5%

    print('Making Technical Indicators Features...')
    TI=TechnicalIndicators(data=df)
    df=TI.fit_transform()

    print('Making Numerical Features...')
    MF=MiscFeatures(data=df)
    df=MF.fit_transform()
    df=format_floats(df)
    df=df.iloc[2:,:]
    print(df.shape)
    print(df.iloc[:1,:7])
    print(df.columns.to_list())

    if save:
        df.to_csv('gold_data_20240529.csv')

    return df

def COMPUTE_TARGET(df: pd.DataFrame) -> pd.DataFrame:
    print('Generating Target...')
    TARGET_GENERATOR=TargetGenerator(data=df,horizon=2,return_threshold=0.02)
    data=TARGET_GENERATOR.generate_targets()
    print(data.TARGET.value_counts()/data.TARGET.value_counts().sum())
    return data


def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    

    print('Training complete')
    return model


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


if __name__ == "__main__":
    
    DATA = pd.read_csv('gold_data_20240529.csv',date_format=True)

    # DATA=CREATE_FEATURES(df=DATA,save=True)

    train_df, test_df = filter_df_by_date(df=DATA, t1='2022-10-20',t2='2023-01-01')
    train_df, test_df = scale_datasets(train=train_df, test=test_df, scaler="standard")
    
    train_df, test_df = COMPUTE_TARGET(train_df), COMPUTE_TARGET(test_df)


    if True:
        print(train_df.shape, test_df.shape)

        TSD=TimeSeriesDataset(data=train_df,sequence_length=21)

        #MODEL=FeedForwardNN(input_dim=16,hidden_dim=64,output_dim=1)
        MODEL=LSTMModel(input_dim=106, hidden_dim=256, output_dim=1, num_layers=2, dropout=0.25)
        LOSS=torch.nn.BCELoss(reduction='mean')
        
        ALPHA=1e-4
        OPTIMIZER=Adam(params=MODEL.parameters(),lr=ALPHA)
        DATALOADER=DataLoader(dataset=TSD,batch_size=8,drop_last=True, shuffle=True)

        MODEL=train_model(model=MODEL,dataloader=DATALOADER,criterion=LOSS,optimizer=OPTIMIZER,num_epochs=20)

        convert_pytorch_to_onnx(model=MODEL, input_size=(8, 21, 106), onnx_file_path='lstm.onnx')

    TEST_DATA=TimeSeriesDataset(data=test_df,sequence_length=21)
    D=DataLoader(dataset=TEST_DATA,batch_size=8,drop_last=True,shuffle=False)
    accuracies=[]
    for inputs, classes in D:
        RESULT=run_inference_onnx(model_path='lstm.onnx',input_data=inputs)
        print(classes)

        print('________________')

        print(RESULT)
        A=accuracy_score(y_true=classes, y_pred=np.where(RESULT > 0.5, 1,0))
        print(A)
        accuracies.append(A)
    
    print(np.array(accuracies).mean())
    print(pd.Series(accuracies).value_counts())

    