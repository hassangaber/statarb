# main.py
import pandas as pd
from rf_dataset import RFDataset
from inference import compute_backtest
from model_training import tune_xgboost_hyperparameters #, tune_lightgbm_hyperparameters

# Load your data
data = pd.read_csv('../assets/data.csv')
data.DATE = pd.to_datetime(data.DATE)

# Define features and target
features = [
    'CLOSE', 'VOLUME', 'HIGH', 'VOLATILITY_90D', 'CLOSE_SMA_3D', 'CLOSE_EWMA_3D', 
    'VOLATILITY_90D_SMA_3D', 'VOLATILITY_90D_EWMA_3D', 'CLOSE_SMA_9D', 'CLOSE_EWMA_9D', 
    'VOLATILITY_90D_SMA_9D', 'VOLATILITY_90D_EWMA_9D', 'CLOSE_SMA_21D', 'CLOSE_EWMA_21D', 
    'VOLATILITY_90D_SMA_21D', 'VOLATILITY_90D_EWMA_21D', 'CLOSE_SMA_50D', 'CLOSE_EWMA_50D', 
    'VOLATILITY_90D_SMA_50D', 'VOLATILITY_90D_EWMA_50D', 'CLOSE_SMA_65D', 'CLOSE_EWMA_65D', 
    'VOLATILITY_90D_SMA_65D', 'VOLATILITY_90D_EWMA_65D', 'CLOSE_SMA_120D', 'CLOSE_EWMA_120D', 
    'VOLATILITY_90D_SMA_120D', 'VOLATILITY_90D_EWMA_120D', 'CLOSE_SMA_360D', 'CLOSE_EWMA_360D', 
    'VOLATILITY_90D_SMA_360D', 'VOLATILITY_90D_EWMA_360D', 'RETURNS', 'CLOSE_ROC_3D', 
    'HIGH_ROC_3D', 'VOLATILITY_90D_ROC_3D', 'CLOSE_ROC_9D', 'HIGH_ROC_9D', 'VOLATILITY_90D_ROC_9D', 
    'CLOSE_ROC_21D', 'HIGH_ROC_21D', 'VOLATILITY_90D_ROC_21D', 'CLOSE_ROC_50D', 'HIGH_ROC_50D', 
    'VOLATILITY_90D_ROC_50D', 'CLOSE_ROC_65D', 'HIGH_ROC_65D', 'VOLATILITY_90D_ROC_65D', 
    'CLOSE_ROC_120D', 'HIGH_ROC_120D', 'VOLATILITY_90D_ROC_120D', 'CLOSE_ROC_360D', 'HIGH_ROC_360D', 
    'VOLATILITY_90D_ROC_360D'
]
target = 'RETURNS'
stock_ids = ['AAPL','NVDA','JNJ']
start_date = '2010-01-01'
end_date = '2023-12-28'
test_start_date = '2024-01-02'

# Create RFDataset instance
rf_dataset = RFDataset(dataframe=data, stock_ids=stock_ids, start_date=start_date, end_date=end_date, features=features, horizon=10, tau=1.0)

# Hyperparameter tuning for XGBoost
xgb_param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
X, y = rf_dataset.get_data()
xgb_best_params = tune_xgboost_hyperparameters(X, y, xgb_param_grid)

# # Hyperparameter tuning for LightGBM
# lgb_param_grid = {
#     'num_leaves': [31, 50, 100],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200, 300]
# }
# lgb_best_params = tune_lightgbm_hyperparameters(X, y, lgb_param_grid)

# Backtest with XGBoost
xgb_model, xgb_accuracy, xgb_X_test, xgb_y_test, xgb_y_pred = compute_backtest(rf_dataset, 'xgboost', xgb_best_params, test_start_date)
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

# Backtest with LightGBM
# lgb_model, lgb_accuracy, lgb_X_test, lgb_y_test, lgb_y_pred = compute_backtest(rf_dataset, 'lightgbm', lgb_best_params, test_start_date)
# print(f"LightGBM Accuracy: {lgb_accuracy:.2f}")
