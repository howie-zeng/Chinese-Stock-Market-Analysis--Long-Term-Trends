
import numpy as np
import optuna
from functools import partial
from tqdm import tqdm
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import concurrent.futures
import os

import parameters as p
import model as m
import re

def calculate_r2_oos(y, y_hat):
    assert len(y_hat) == len(y)
    SSR = ((y - y_hat)**2).sum()
    SST = (y**2).sum()
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
        r_2 = calculate_r2_oos(y_val, predictions)
        return r_2
        
    best_trials = {}   
    for num_layers in tqdm(num_layers_range, desc="Tuning Models"):
        model_name = f'NNModel_nn{num_layers}'
        print(f"Tunning {num_layers}")
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
                # Learning rate, expanded upper limit for broader exploration
                'eta': trial.suggest_float('eta', 1e-8, 0.2, log=True),
                # Minimum loss reduction required to make a further partition
                'gamma': trial.suggest_float('gamma', 0.1, 1),
                # Number of trees in the ensemble
                'n_estimators': trial.suggest_int("n_estimators", 10, 300),
                # Subsample ratio of the training instances
                'subsample': trial.suggest_float('subsample', 0.75, 1),
                # Number of parallel trees in the ensemble (for boosting types that support it)
                'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 10),
                 # Subsample ratio of columns when constructing each tree
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 1),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.75, 1),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 0.8),
                # Maximum depth of a tree, expanded range for complex interactions
                'max_depth': trial.suggest_int('max_depth', 8, 30),
                # Minimum sum of instance weight (hessian) needed in a child
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                # L1 regularization term on weights
                'lambda': trial.suggest_float('lambda', 1e-8, 5, log=True),
                # L2 regularization term on weights
                'alpha': trial.suggest_float('alpha', 1e-8, 5, log=True),
                # Objective function to minimize, including options more robust to noise
                'objective': trial.suggest_categorical('objective', ['reg:squarederror', 'reg:pseudohubererror']),
                # Maximum number of leaves; use it with 'lossguide' grow policy
                'max_leaves': trial.suggest_int('max_leaves', 0, 256),
            }
        else:
            model_instance = model_class(params=params)

        model_instance = model_class(params=params)
        if 'NNModel' in model_name:
            model_fitted = m.train(X_train, y_train, model_instance, X_val, y_val)
        else:
            model_fitted = m.train(X_train, y_train, model_instance)
        predictions = m.test(X_val, model_fitted)
        r_2 = calculate_r2_oos(y_val, predictions)
        return r_2
        

    best_trials = {}
    studies = {}
    for model_class in tqdm(model_classes, desc="Tuning Models"):
        model_name = model_class().name if hasattr(model_class(), "name") else model_class().__class__.__name__
        model_class.name = model_name
        print(f"Tunning {model_name}")
        study = optuna.create_study(direction="maximize")
        if model_name == "XGBoostModel":
            n_jobs = 1
        else:
            n_jobs = -1
        objective_with_args = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, model_class=model_class, model_name=model_name)
        study.optimize(objective_with_args, n_trials=n_trials, n_jobs=n_jobs)
        best_trials[model_name] = study.best_trial.params
        studies[model_name] = study

    for model_name, params in best_trials.items():
        print(f"Best trial for {model_name}:", params)

    return best_trials, studies

def calculate_feature_importance(model_classes, X_train, y_train, X_val, y_val, features, permutation_importance=False):
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
        i = 1
        for variable in features:
            progress_bar.set_postfix({"Model Name": model_name, "Variable": variable, "n": f"{i}/{len(features)}"})
            X_modified = X_train_scaled.copy()
            if permutation_importance:
                X_modified[variable] = np.random.permutation(X_modified[variable])
            else:
                X_modified[variable] = 0
            modified_r2 = calculate_r2_oos(y_train, model_class.predict(X_modified))
            reduction = original_r2 - modified_r2
            model_changes.append(reduction)
            i += 1
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

def fit_and_evaluate_model(model_instance, params, X_train, y_train, X_val, y_val):
    model_instance.fit(X_train, y_train)
    train_score = calculate_r2_oos(y_train, model_instance.predict(X_train))
    val_score = calculate_r2_oos(y_val, model_instance.predict(X_val))
    return {
        **params,
        'train_score': train_score,
        'val_score': val_score
    }

def explore_parameter_grid(model_class, param_grid, X_train, y_train, X_val, y_val, multiprocess=True):
    results = []
    params_list = list(ParameterGrid(param_grid))
    if not multiprocess:
        for params in tqdm(params_list):
            result = fit_and_evaluate_model(model_class(**params), params, X_train, y_train, X_val, y_val)
            results.append(result)
    else:
        X_train.setflags(write=True)
        y_train.setflags(write=True)
        X_val.setflags(write=True)
        y_val.setflags(write=True)
        num_workers = os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(fit_and_evaluate_model, model_class(**params), params, X_train, y_train, X_val, y_val)
                for params in params_list
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Exploring Parameters"):
                result = future.result()
                if result: 
                    results.append(result)
    results_df = pd.DataFrame(results)
    return results_df

