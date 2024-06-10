
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def format_floats(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float"]).columns
    for col in float_cols:
        df[col] = df[col].map("{:.6f}".format).astype(np.float64)
    return df


def filter_df_by_date(df: pd.DataFrame, t1:str, t2: str, t0: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    make_date = lambda x: pd.to_datetime(x)
    df.DATE = pd.to_datetime(df.DATE)

    assert make_date(t1) < make_date(t2)

    if t0 is not None:
        train = df.loc[(df.DATE >= make_date(t0)) & (df.DATE < make_date(t1))]
        test = df.loc[df.DATE >= make_date(t2)]
    else:
        train = df.loc[df.DATE < make_date(t1)]
        test = df.loc[df.DATE >= make_date(t2)]

    return (train, test)

def handle_infinite_values(df:pd.DataFrame):
    return df.replace([np.inf, -np.inf], np.nan)

def fill_missing_values(df:pd.DataFrame):
    return df.fillna(df.median())

def scale_datasets(train: pd.DataFrame, test: pd.DataFrame, scaler: str = "standard") -> tuple[pd.DataFrame, pd.DataFrame]:
    # Identify the numerical columns
    numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    non_numerical = list(set(train.columns.to_list()) - set(numerical_cols))
    print(non_numerical)

    train_ = handle_infinite_values(train[numerical_cols])
    test_ = handle_infinite_values(test[numerical_cols])
    
    train_ = fill_missing_values(train_)
    test_ = fill_missing_values(test_)

    if scaler == "standard":
        scaler_transformer = StandardScaler(with_mean=True, with_std=True).set_output(transform='pandas')
    elif scaler == "minmax":
        scaler_transformer = MinMaxScaler(feature_range=(-1, 1)).set_output(transform='pandas')

    # preprocessor = ColumnTransformer(
    #     transformers=[('num', scaler_transformer, numerical_cols)],
    #     remainder='passthrough'
    # ).set_output(transform='pandas')
    
    train_scaled = scaler_transformer.fit_transform(train_)
    test_scaled = scaler_transformer.transform(test_)

    train_scaled=pd.concat([train_scaled,train[non_numerical]],axis=1)
    test_scaled=pd.concat([test_scaled,test[non_numerical]],axis=1)

    return train_scaled, test_scaled 
