# model_training.py
import xgboost as xgb
# import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

def train_xgboost_model(X_train, y_train, params=None):
    model = xgb.XGBClassifier(**params) if params else xgb.XGBClassifier()
    model.fit(X_train, y_train)
    return model

# def train_lightgbm_model(X_train, y_train, params=None):
#     model = lgb.LGBMClassifier(**params) if params else lgb.LGBMClassifier()
#     model.fit(X_train, y_train)
#     return model

def tune_xgboost_hyperparameters(X, y, param_grid, cv=3):
    model = xgb.XGBClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_params_

# def tune_lightgbm_hyperparameters(X, y, param_grid, cv=3):
#     model = lgb.LGBMClassifier()
#     grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
#     grid_search.fit(X, y)
#     return grid_search.best_params_
