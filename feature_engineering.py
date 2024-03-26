import pandas as pd
import numpy as np
import parameters as p
import statsmodels.api as sm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def calculate_momentum(data_monthly):
    data_monthly.sort_index(inplace=True)
    group = data_monthly.groupby(p.stockID)['return_monthly']

    data_monthly['mom1m'] = group.transform(lambda x: x.shift(1) ) 
    data_monthly['mom12m'] = group.transform(lambda x: x.shift(2).rolling(window=11).sum()) 
    data_monthly['mom6m'] = group.transform(lambda x: x.shift(2).rolling(window=5).sum())  
    data_monthly['mom36m'] = group.transform(lambda x: x.shift(14).rolling(window=24).sum())  

    return data_monthly
def calculate_weekly_returns(df, price_col):
    weekly_prices = df[price_col].resample('W').last()
    weekly_returns = weekly_prices.pct_change().dropna()
    return weekly_returns

def process_ticker_data(args):
    ticker_data, market_col, stock_col, name = args
    ticker_data.set_index('date', inplace=True)

    stock_weekly_returns = calculate_weekly_returns(ticker_data, stock_col)
    market_weekly_returns = calculate_weekly_returns(ticker_data, market_col)
    
    combined_weekly_returns = pd.DataFrame({
        'stock_returns': stock_weekly_returns,
        'market_returns': market_weekly_returns
    }).dropna()
    
    results = []
    if len(combined_weekly_returns) > 1:
        X = sm.add_constant(combined_weekly_returns[['market_returns']])
        y = combined_weekly_returns['stock_returns']
        
        for i in range(52, len(y)):
            y_temp = y.iloc[max(i-156, 0):i-1]
            X_temp = X.iloc[max(i-156, 0):i-1]
            model = sm.OLS(y_temp, X_temp).fit()
            
            result = {
                'date': y_temp.index[-1],
                'alpha': model.params[0],
                'beta': model.params[1],
                p.stockID: name
            }
            results.append(result)
    return results

def calculate_stock_level_alpha_and_beta(data, market_col='000905_close', stock_col='close_adj'):
    if 'date' not in data.columns:
        raise ValueError("Data should have 'date'.")
    ticker_groups = data.groupby(p.stockID)
    process_args = [(group.copy(), market_col, stock_col, name) for name, group in ticker_groups]
    
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_ticker_data, arg) for arg in process_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating alpha and beta"):
            result = future.result()
            results.extend(result)
    results_df = pd.DataFrame(results)
    merged_data = pd.merge_asof(data, results_df.sort_values('date'), on='date', by=p.stockID, tolerance=pd.Timedelta(days=5), direction='backward')
    return merged_data
        
def feature_construction(data_daily, data_monthly): 
    # daily
    data_daily['month'] = data_daily['date'].dt.to_period('m')
    maxret = data_daily.groupby([p.stockID, 'month'])['return_daily'].max().reset_index()
    maxret = maxret.rename(columns={'return_daily': 'maxret'})
    data_daily = pd.merge(data_daily, maxret, left_on=[p.stockID, 'month'], right_on=[p.stockID, 'month'], how='left')

    # volatility = data_daily.groupby([p.stockID, 'month'])['return_daily'].std().rename(columns={'return_daily': 'volatility'})
    # data_daily = pd.merge(data_daily, volatility, left_on=[p.stockID, 'month'], right_on=[p.stockID, 'month'], how='left')

    # monthly
    chmom_6m = data_monthly.groupby(p.stockID)['return_monthly'].rolling(window=6).apply(lambda x: (1+x).prod() - 1, raw=True).shift(1).reset_index(level=0, drop=True)
    chmom_12m = data_monthly.groupby(p.stockID)['return_monthly'].rolling(window=6).apply(lambda x: (1+x).prod() - 1, raw=True).shift(7).reset_index(level=0, drop=True)
    data_monthly['chmom'] = chmom_6m - chmom_12m
    data_monthly = calculate_momentum(data_monthly)

    data_daily.drop('month', axis=1, inplace=True)
    data_daily.set_index('date', inplace=True)
    data_monthly.set_index('date', inplace=True)
    return data_daily, data_monthly