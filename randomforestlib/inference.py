# inference.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from model_training import train_xgboost_model #, train_lightgbm_model

def predict_model(model:callable, X:pd.DataFrame)->np.array:
    print(model)
    print(X)
    return model.predict(X)

def compute_backtest(rf_dataset, model_type='xgboost', params=None, test_start_date='2024-01-02'):
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test = rf_dataset.train_test_split_and_scale(test_start_date)
    
    # Train the model
    if model_type == 'xgboost':
        model = train_xgboost_model(X_train, y_train, params)
    # elif model_type == 'lightgbm':
    #     model = train_lightgbm_model(X_train, y_train, params)
    else:
        raise ValueError("Invalid model type. Choose 'xgboost' or 'lightgbm'.")
    
    # Make predictions
    y_pred = predict_model(model, X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred
