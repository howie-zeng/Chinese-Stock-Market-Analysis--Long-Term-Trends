
import numpy as np
import optuna
from functools import partial
from tqdm import tqdm


import parameters as p
import model as m

def calculate_r2_oos(y, y_hat):
    assert len(y_hat) == len(y)
    n = len(y_hat)
    mean_hat = np.mean(y_hat)
    SSR = np.sum((y - mean_hat)**2)
    SST = np.sum(y)
    res = 1 - SSR/SST
    return res

def hyperparameter_tuning(data, model_classes, n_trials=100):
    def objective(trial, data, model_class, model_name):
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
        model_fitted, scaler = m.train(data, model_instance)
        validation_res = m.validation(data, model_fitted, scaler)
        r_2 = calculate_r2_oos(validation_res['Y'], validation_res['prediction'])
        return r_2
        

    best_trials = {}
    for model_class in tqdm(model_classes, desc="Tuning Models"):
        model_name = model_class().name if hasattr(model_class(), "name") else model_class().__class__.__name__
        print(f"Tunning {model_name}")
        study = optuna.create_study(direction="maximize")
        objective_with_args = partial(objective, data=data, model_class=model_class, model_name=model_name)
        study.optimize(objective_with_args, n_trials=n_trials)
        best_trials[model_name] = study.best_trial.params

    for model_name, params in best_trials.items():
        print(f"Best trial for {model_name}:", params)

    return best_trials
