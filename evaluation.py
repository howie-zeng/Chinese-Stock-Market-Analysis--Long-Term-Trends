
import numpy as np
import optuna
from functools import partial
from tqdm import tqdm
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.preprocessing import StandardScaler

import parameters as p
import model as m
import re

def calculate_r2_oos(y, y_hat):
    assert len(y_hat) == len(y)
    SSR = np.sum((y - y_hat)**2)
    SST = np.sum(y)
    res = 1 - SSR/SST
    return res

def nn_tunning(num_layers_range, X_train, y_train, X_val, y_val, n_trials=100):
    np.random.seed(p.RandomState)
    def objective(trial, X_train, y_train, X_val, y_val, num_layers):
        params ={
            'architecture': [32, 16, 8, 4, 2][:num_layers],
            'output_dim': 1,
            'loss': trial.suggest_categorical('objective', ['mse', 'huberloss']),
            'lambda': trial.suggest_float('lambda_value', 1e-5, 1e-2, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 512, 2048, 10000])
        }
        model_instance = m.NNModel(params=params, input_dim=X_train.shape[1], num_layers=num_layers)
        model_fitted, scaler = m.train(X_train, y_train, model_instance, X_val, y_val)
        predictions = m.test(X_val, model_fitted, scaler)
        r_2 = calculate_r2_oos(y_val.values, predictions)
        return r_2
        
    best_trials = {}   
    for num_layers in tqdm(num_layers_range, desc="Tuning Models"):
        model_name = f'NNModel_nn{num_layers}'
        print(f"Tunning NN{num_layers}")
        study = optuna.create_study(direction="maximize")
        objective_with_args = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, num_layers=num_layers)
        study.optimize(objective_with_args, n_trials=n_trials, n_jobs=-1)
        best_trials[model_name] = study.best_trial.params
    
    return best_trials
    

def hyperparameter_tuning(X_train, y_train, X_val, y_val, model_classes, n_trials=100):
    np.random.seed(p.RandomState)
    def objective(trial, X_train, y_train, X_val, y_val, model_class, model_name):
        params = {}
        if model_name == 'OLSModel' or model_name == 'OLS3Model':
            params = {
                'epsilon': trial.suggest_float("epsilon", 1.0, 5),
                'alpha': trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                'max_iter': 1000
            }
        elif model_name == 'PLSModel':
            params = {
                'n_components': trial.suggest_int("n_components", 2, len(X_train.columns)),
                'max_iter': 1000
            }
        elif model_name == 'LASSOModel':
            params = {
                'alpha': trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                'max_iter': 1000
            }
        elif model_name== 'ElasticNetModel':
            params = {
                'alpha': trial.suggest_float("alpha", 1e-4, 1.0, log=True), 
                'l1_ratio': trial.suggest_float("l1_ratio", 0.0, 1.0),
                'max_iter': 1000
            }
        elif model_name == 'GBRTModel':
            params = {
                'loss': trial.suggest_categorical("loss", ["huber", "squared_error", "quantile"]),
                'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int("max_depth", 3, 10),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 10),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
                'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"])
            }
            if params['loss'] in ["huber", "quantile"]:
                params['alpha'] = trial.suggest_float("alpha", 1e-4, 1.0, log=True)

        elif model_name == 'RFModel':
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 50, 1000),
                'criterion': trial.suggest_categorical('criterion', ["squared_error", "friedman_mse"]),
                'max_depth': trial.suggest_int("max_depth", 5, 30),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 10),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
                'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
        elif model_name == 'XGBoostModel':
            params = {
                'booster': trial.suggest_categorical('booster', ['gbtree']),
                'eta': trial.suggest_float('eta', 0.01, 0.3),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                'subsample': trial.suggest_float('subsample', 0.5, 1),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1),
                'objective': trial.suggest_categorical('objective', ['reg:squarederror', 'reg:pseudohubererror']),
                'n_estimators': trial.suggest_int("n_estimators", 50, 1000),
            }
        else:
            model_instance = model_class(params=params)
        
        model_instance = model_class(params=params)
        if 'NNModel' in model_name:
            model_fitted, scaler = m.train(X_train, y_train, model_instance, X_val, y_val)
        else:
            model_fitted, scaler = m.train(X_train, y_train, model_instance)
        predictions = m.test(X_val, model_fitted, scaler)
        r_2 = calculate_r2_oos(y_val.values, predictions)
        return r_2
        

    best_trials = {}
    for model_class in tqdm(model_classes, desc="Tuning Models"):
        model_name = model_class().name if hasattr(model_class(), "name") else model_class().__class__.__name__
        model_class.name = model_name
        print(f"Tunning {model_name}")
        study = optuna.create_study(direction="maximize")
        objective_with_args = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, model_class=model_class, model_name=model_name)
        study.optimize(objective_with_args, n_trials=n_trials, n_jobs=-1)
        best_trials[model_name] = study.best_trial.params

    for model_name, params in best_trials.items():
        print(f"Best trial for {model_name}:", params)

    return best_trials

def feature_importance(model_classes, X_train, y_train, X_val, y_val, features, permutation_importance=False):
    importance_dict = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    X_val_scaled = scaler.transform(X_val)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    progress_bar = tqdm(model_classes, desc="Calculating Feature Importance")
    for model_class in progress_bar:
        model_name = model_class.name if hasattr(model_class, "name") else model_class.__class__.__name__
        progress_bar.set_postfix({"Model Name": model_name})
        if 'NNModel' in model_name:
            model_class.fit(X_train_scaled, y_train, X_val_scaled, y_val)
        else:
            model_class.fit(X_train_scaled, y_train)
        original_r2 = calculate_r2_oos(y_train, model_class.predict(X_train_scaled))
        model_changes = []
        for variable in features:
            X_modified = X_train_scaled.copy()
            if permutation_importance:
                X_modified[variable] = np.random.permutation(X_modified[variable])
            else:
                X_modified[variable] = 0

            modified_r2 = calculate_r2_oos(y_train, model_class.predict(X_modified))
            reduction = original_r2 - modified_r2
            model_changes.append(reduction)
        importance_dict[model_name] = model_changes
    importance_df = pd.DataFrame(importance_dict, index=features)
    return importance_df

def sort_into_deciles(predictions):
    return predictions.groupby(predictions.index).transform(lambda x: pd.qcut(x, 10, labels=False)).values + 1

def form_portfolios(model_res, market_cap):
    stock_returns = model_res['y']
    performances = {}

    for model_name in model_res.drop(columns=['y']).columns:
        deciles = sort_into_deciles(model_res[model_name])
        decile_weights = market_cap.groupby(deciles).apply(lambda x: x / x.sum())

        long_short_returns = (
            (stock_returns[deciles == 10] * decile_weights[deciles == 10]).sum() -
            (stock_returns[deciles == 1] * decile_weights[deciles == 1]).sum()
        )

        long_only_returns = (stock_returns[deciles == 10] * decile_weights[deciles == 10]).sum()

        performances[model_name] = {
            'long_short': long_short_returns,
            'long_only': long_only_returns
        }
    return performances

def calculate_performance_statistics(portfolio_returns):
    statistics = {}
    for model_name, returns in portfolio_returns.items():
        avg_return = np.mean(returns)
        std = np.std(returns)
        sharpe_ratio = avg_return / std if std != 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(returns) - returns)  # Simplified calculation
        max_1m_loss = np.min(returns)

        statistics[model_name] = {
            'Avg': avg_return * 100,
            'Std': std * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown * 100,
            'Max 1M Loss': max_1m_loss * 100
        }
    return pd.DataFrame(statistics).T


