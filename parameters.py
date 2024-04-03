import os


offset = 252
history_length = 252*7

offset = 252
RandomState = 2024

stockID='Ticker'

currentPath = os.getcwd()
dataPath = currentPath + "\\data"
econDataPath = dataPath + "\\econ"
cleanedDataPath = dataPath + "\\cleaned"
merged_data_daily = os.path.join(cleanedDataPath, "merged_data_daily.csv")
merged_data_monthly = os.path.join(cleanedDataPath, "merged_data_monthly.csv")
zz_data_daily = os.path.join(cleanedDataPath, "zz_data_daily.csv")
zz_data_monthly = os.path.join(cleanedDataPath, "zz_data_monthly.csv")

daily_files = ['close.csv', 'close_adj.csv', "beta_000905.csv", 'daily_ret_vol_roll_126.csv', 'return_daily.csv', 'total_market_value.csv', 'turnover_daily.csv']
monthly_files = ['illiquidity_monthly.csv', 'mve_log.csv', 'return_monthly.csv', 'ret_vol_monthly.csv', 'std_dolvol_monthly.csv', 'std_turnover_monthly.csv', 'zero_trade_days.csv']
sector_files_daily = ['000905.csv', '000905_return_daily.csv']
sector_files_monthly = ['000905_return_monthly.csv']
dataFiles = [f for f in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath, f))]
missing_files = [file for file in dataFiles if file not in daily_files + monthly_files + sector_files_daily + sector_files_monthly]

ols3_predictors = ['mve_log', 'mom1m'] #size, book-to-market and momentum
# Parameters 
ols_params = {'epsilon': 4.99989279171685, 'alpha': 0.8458067503168284}
ols3_params = {'epsilon': 4.999826064384155, 'alpha': 0.07030718705181585}
pls_params = {'n_components': 7}
lasso_params = {'alpha': 0.0024505937395793687}
elasticnet_params = {'alpha': 0.004456327578256146, 'l1_ratio': 0.5524035315018496}

gbrt_params = {
}

rf_params = {
}

xgboost_params = {'eta': 0.14315876654070525, 'gamma': 0.15267525903299634, 'n_estimators': 259, 'subsample': 0.816012056487514, 'num_parallel_tree': 7, 'colsample_bytree': 0.7626462759013348, 'colsample_bylevel': 0.8006154025732285, 'colsample_bynode': 0.5140560102311738, 'max_depth': 9, 'min_child_weight': 6, 'lambda': 1.0107283039394311e-08, 'alpha': 0.2428371143422945, 'objective': 'reg:squarederror', 'max_leaves': 164}

vasa_params = {
}

nn_params = {
}


