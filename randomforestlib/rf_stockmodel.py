import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import pickle
from typing import List, Tuple, Optional, Union

class StockModel:
    def __init__(self, dataframe: pd.DataFrame, stock_ids: List[str], start_date: str, end_date: str, features: List[str], horizon: int = 4, tau: float = 1e-6):
        self.data = dataframe
        self.stock_ids = stock_ids
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.features = features
        self.horizon = horizon
        self.tau = tau
        self.processed_data = pd.DataFrame()
        self.preprocess()

    def preprocess(self):
        processed_data_list = []
        for stock_id in self.stock_ids:
            stock_data = self.data[(self.data['ID'] == stock_id)]
            stock_data['future_returns'] = stock_data['RETURNS'].diff(periods=self.horizon)
            stock_data['momentum'] = stock_data['RETURNS'].rolling(window=self.horizon).mean()
            threshold = self.tau * stock_data['VOLATILITY_90D']
            stock_data.dropna(inplace=True)
            threshold = threshold.reindex(stock_data.index)
            stock_data['future_returns'] = stock_data['future_returns'].reindex(stock_data.index)
            conditions = [
                (stock_data['future_returns'] > threshold) & (stock_data['momentum'] > 0) & (stock_data["CLOSE_SMA_9D"] > stock_data["CLOSE_SMA_21D"]),
                (stock_data['future_returns'] < -threshold) & (stock_data['momentum'] < 0) & (stock_data["CLOSE_SMA_9D"] < stock_data["CLOSE_SMA_21D"])
            ]
            choices = [1, -1]
            stock_data['label'] = np.select(conditions, choices, default=0)
            stock_data['label'] = stock_data['label'].map({-1: 0, 0: 1, 1: 2})
            processed_data_list.append(stock_data)
        self.processed_data = pd.concat(processed_data_list).reset_index(drop=True)

    def train_test_split_and_scale(self, test_start_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        test_start_date = pd.to_datetime(test_start_date)
        train_data = self.processed_data[self.processed_data['DATE'] < test_start_date]
        test_data = self.processed_data[self.processed_data['DATE'] >= test_start_date]
        X_train = train_data[self.features]
        y_train = train_data['label']
        X_test = test_data[self.features]
        y_test = test_data['label']
        scaler = StandardScaler().set_output(transform='pandas')
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train, y_train = self.downsample_random(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def downsample_random(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.concat([X, y.reset_index(drop=True)], axis=1)
        min_count = df['label'].value_counts().min()
        df_downsampled = df.groupby('label').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
        X_downsampled = df_downsampled[self.features]
        y_downsampled = df_downsampled['label']
        return X_downsampled, y_downsampled

    def tune_hyperparameters(self, model_type: str, param_grid: dict, cv: int = 3) -> dict:
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
        else:
            raise ValueError("Invalid model type. Choose 'xgboost' or 'lightgbm'.")
        tscv = TimeSeriesSplit(n_splits=cv)
        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='f1_macro',n_jobs=-1, verbose=3)
        X, y = self.get_data()
        grid_search.fit(X, y)
        return grid_search.best_params_
    
    def save_xgboost_model(self, model, file_path='../assets/xgb.pkl'):
        """
        Save the XGBoost model to a pickle file.

        Parameters:
        model (xgb.Booster): The XGBoost model to save.
        file_path (str): The file path where the model will be saved.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    def get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        X = self.processed_data[self.features]
        y = self.processed_data['label']
        return X, y

    def fit(self, model_type: str = 'xgboost', params: Optional[dict] = None, test_start_date: str = '2024-01-02') -> Tuple[Union[xgb.XGBClassifier, None], float, pd.DataFrame, pd.Series, np.ndarray]:
        X_train, X_test, y_train, y_test = self.train_test_split_and_scale(test_start_date)
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**params) if params else xgb.XGBClassifier()
        else:
            raise ValueError("Invalid model type. Choose 'xgboost' or 'lightgbm'.")
        model.fit(X_train, y_train)
        self.save_xgboost_model(model=model)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy, X_test, y_test, y_pred

    def backtest(self, model_type: str = 'xgboost', params: Optional[dict] = None, test_start_date: str = '2024-01-02') -> Tuple[Union[xgb.XGBClassifier, None], float, pd.DataFrame, pd.Series, np.ndarray]:
        model, accuracy, X_test, y_test, y_pred = self.fit(model_type, params, test_start_date)
        return model, accuracy, X_test, y_test, y_pred

if __name__ == "__main__":

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
        'VOLATILITY_90D_SMA_360D', 'VOLATILITY_90D_EWMA_360D', 'CLOSE_ROC_3D', 
        'HIGH_ROC_3D', 'VOLATILITY_90D_ROC_3D', 'CLOSE_ROC_9D', 'HIGH_ROC_9D', 'VOLATILITY_90D_ROC_9D', 
        'CLOSE_ROC_21D', 'HIGH_ROC_21D', 'VOLATILITY_90D_ROC_21D', 'CLOSE_ROC_50D', 'HIGH_ROC_50D', 
        'VOLATILITY_90D_ROC_50D', 'CLOSE_ROC_65D', 'HIGH_ROC_65D', 'VOLATILITY_90D_ROC_65D', 
        'CLOSE_ROC_120D', 'HIGH_ROC_120D', 'VOLATILITY_90D_ROC_120D', 'CLOSE_ROC_360D', 'HIGH_ROC_360D', 
        'VOLATILITY_90D_ROC_360D'
    ]
    stock_ids = ['AAPL', 'NVDA', 'JNJ']
    start_date = '2010-01-01'
    end_date = '2023-12-28'
    test_start_date = '2024-01-02'

    stock_model = StockModel(dataframe=data, stock_ids=stock_ids, start_date=start_date, end_date=end_date, features=features, horizon=4,tau=1e-6)

    xgb_param_grid = {
        'max_depth': [5,7,10],
        'learning_rate': [0.0005, 0.005, 0.009],
        'n_estimators': [50,75,100]
    }
    xgb_best_params = stock_model.tune_hyperparameters('xgboost', xgb_param_grid)

    # Backtest with XGBoost
    xgb_model, xgb_accuracy, xgb_X_test, xgb_y_test, xgb_y_pred = stock_model.backtest('xgboost', xgb_best_params, test_start_date)
    print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

