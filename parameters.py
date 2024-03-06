# Parameters for OLS
ols_params = {
    "fit_intercept": [True, False]
}

# Parameters for LASSO
lasso_params = {
    "alpha": [0.1, 0.5, 1.0],
    "fit_intercept": [True, False]
}

# Parameters for ElasticNet
elasticnet_params = {
    "alpha": [0.1, 0.5, 1.0],
    "l1_ratio": [0.1, 0.5, 0.9],
    "fit_intercept": [True, False]
}

# Parameters for GBRT
gbrt_params = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2]
}

# Parameters for RF
rf_params = {
    "n_estimators": [50, 100, 200]
}

# Parameters for XGBoost
xgboost_params = {
    "n_jobs": -1
}

# Parameters for NN
nn_params = {
    "architecture": [[32], [64, 64], [128, 128]],
    "epochs": [10, 20, 50],
    "batch_size": [16, 32, 64]
}
