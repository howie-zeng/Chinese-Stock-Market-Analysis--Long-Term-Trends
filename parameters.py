import os

offset = 252
history_length = 252*7

offset = 252

stockID='Ticker'

currentPath = os.getcwd()
dataPath = currentPath + "\\data"
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

ols3_predictors = ['return_monthly', 'mve_log', 'mom1m'] #size, book-to-market and momentum
# Parameters 
ols_params = {
}

ols3_params = { 
}

pls_params = {

}

lasso_params = {
}

elasticnet_params = {
}

gbrt_params = {
}

rf_params = {
}

xgboost_params = {
}

vasa_params = {
}

nn_params = {
}


