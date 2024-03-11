offset = 252
history_length = 252*7
training_sample = ['2012-01-01', '2015-12-31']
validation_sample = ['2016-01-01', '2018-12-31']
testing_sample= ['2019-01-01', '2024-12-31']


# Parameters for OLS
ols_params = {
    "n_jobs": -1
}

# Parameters for LASSO
lasso_params = {
}
# Parameters for ElasticNet
elasticnet_params = {
}

# Parameters for GBRT
gbrt_params = {
}

# Parameters for RF
rf_params = {
    "n_jobs": -1
}

# Parameters for XGBoost
xgboost_params = {
    "n_jobs": -1
}

# Parameters for NN
nn_params = {
}
