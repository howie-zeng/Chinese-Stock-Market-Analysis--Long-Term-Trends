import os

offset = 252
history_length = 252*7

offset = 252


stockID='Ticker'

training_sample = ['2012-01-01', '2019-12-31']
validation_sample = ['2020-01-01', '2022-06-30']
testing_sample= ['2022-07-01', '2024-12-31']

currentPath = os.getcwd()
dataPath = currentPath + "\\data"
cleanedDataPath = dataPath + "\\cleaned"
merged_data_daily = os.path.join(cleanedDataPath, "merged_data_daily.csv")
merged_data_monthly = os.path.join(cleanedDataPath, "merged_data_monthly.csv")

daily_files = ['close.csv', 'close_adj.csv', "beta_000905.csv", 'daily_ret_vol_roll_126.csv', 'return_daily.csv', 'total_market_value.csv', 'turnover_daily.csv']
monthly_files = ['illiquidity_monthly.csv', 'mve_log.csv', 'return_monthly.csv', 'ret_vol_monthly.csv', 'std_dolvol_monthly.csv', 'std_turnover_monthly.csv', 'zero_trade_days.csv']
sector_files = ['000905.csv', '000905_return_daily.csv', '000905_return_monthly.csv']
dataFiles = [f for f in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath, f))]
missing_files = [file for file in dataFiles if file not in daily_files + monthly_files + sector_files]
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


