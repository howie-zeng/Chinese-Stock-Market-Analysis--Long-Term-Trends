
import numpy as np
import optuna
from functools import partial
from tqdm import tqdm
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.preprocessing import StandardScaler

import parameters as p
import model as m

def calculate_r2_oos(y, y_hat):
    assert len(y_hat) == len(y)
    SSR = np.sum((y - y_hat)**2)
    SST = np.sum(y)
    res = 1 - SSR/SST
    return res

def hyperparameter_tuning(X_train, y_train, X_val, y_val, model_classes, n_trials=100):
    def objective(trial, X_train, y_train, X_val, y_val, model_class, model_name):
        params = {}
        if model_name == 'OLSModel' or model_name == 'OLS3Model':
            epsilon = trial.suggest_float("epsilon", 1.1, 2.0)
            alpha = trial.suggest_loguniform("alpha", 1e-4, 1.0)
            params = {'epsilon': epsilon, 'alpha': alpha}
            
        elif model_name == 'PLSModel':
            n_components = trial.suggest_int("n_components", 2, 10)
            params = {'n_components': n_components}
            
        elif model_name == 'LASSOModel':
            alpha = trial.suggest_loguniform("alpha", 1e-4, 1.0)
            params = {'alpha': alpha}
            
        elif model_name== 'ElasticNetModel':
            alpha = trial.suggest_loguniform("alpha", 1e-4, 1.0)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            params = {'alpha': alpha, 'l1_ratio': l1_ratio}
            
        elif model_name == 'GBRTModel':
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.3)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            loss = trial.suggest_categorical("loss", ["huber"])
            params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth,
                    'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'loss': loss}
            
        elif model_name == 'RFModel':
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            max_depth = trial.suggest_int("max_depth", 5, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 
                    'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
            
        elif model_name == 'XGBoostModel':
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.3)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
            gamma = trial.suggest_float("gamma", 0, 0.5)
            params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth,
                    'min_child_weight': min_child_weight, 'gamma': gamma}
        
        # #incomplete
        # if model_class.__name__ == 'NNModel':
        #     # Handle NNModel-specific parameters, such as architecture
        #     num_layers = trial.suggest_int('num_layers', 1, 5)
        #     architecture = [trial.suggest_int(f'units_layer_{i}', 32, 256) for i in range(num_layers)]
        #     params.update({
        #         "architecture": architecture,
        #         "output_dim": 1,  # Assuming a fixed output dimension
        #         # Add any NNModel specific tunings, such as learning rate if not globally defined
        #     })
        #     model_instance = model_class(params=params, input_dim=input_dim, num_layers=num_layers)
        # else:
        #     model_instance = model_class(params=params)
        
        model_instance = model_class(params=params)
        model_fitted, scaler = m.train(X_train, y_train, model_instance)
        validation_res = m.validation(X_val, y_val, model_fitted, scaler)
        r_2 = calculate_r2_oos(validation_res['y'], validation_res['y_pred'])
        return r_2
        

    best_trials = {}
    for model_class in tqdm(model_classes, desc="Tuning Models"):
        model_name = model_class().name if hasattr(model_class(), "name") else model_class().__class__.__name__
        print(f"Tunning {model_name}")
        study = optuna.create_study(direction="maximize")
        objective_with_args = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, model_class=model_class, model_name=model_name)
        study.optimize(objective_with_args, n_trials=n_trials)
        best_trials[model_name] = study.best_trial.params

    for model_name, params in best_trials.items():
        print(f"Best trial for {model_name}:", params)

    return best_trials

def feathre_importance(model_classes, X_train, y_train, features, permutation_importance=False):
    # change to perform modified versio  30 times
    importance_dict = {}
    percentage_change_dict = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    
    for model_class in tqdm(model_classes, desc="Calculating Feature Importance"):
        model_name = model_class.name if hasattr(model_class, "name") else model_class.__class__.__name__
        model_changes = []
        percentage_changes = []
        
        model_class.fit(X_train_scaled, y_train)
        original_r2 = calculate_r2_oos(y_train.values, model_class.predict(X_train_scaled))

        for variable in features:
            X_modified = X_train_scaled.copy()
            if permutation_importance:
                X_modified[variable] = np.random.permutation(X_modified[variable].values)
            else:
                X_modified[variable] = 0
            
            model_class.fit(X_modified, y_train)
            modified_r2 = calculate_r2_oos(y_train.values, model_class.predict(X_modified))
            
            reduction = original_r2 - modified_r2 
            percentage_change = reduction/original_r2 * 100

            model_changes.append(reduction)
            percentage_changes.append(percentage_change)
        
        importance_dict[model_name] = model_changes
        percentage_change_dict[model_name] = percentage_changes

    percentage_change_df = pd.DataFrame(percentage_change_dict, index=features)
    importance_df = pd.DataFrame(importance_dict, index=features)
    return importance_df, percentage_change_df

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


