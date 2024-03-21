import pandas as pd
import numpy as np
import parameters as p

def calculate_momentum(data_monthly):
    data_monthly.sort_index(inplace=True)
    group = data_monthly.groupby(p.stockID)['return_monthly']

    data_monthly['mom1m'] = group.transform(lambda x: x.shift(1) ) 
    data_monthly['mom12m'] = group.transform(lambda x: x.shift(2).rolling(window=11).sum()) 
    data_monthly['mom6m'] = group.transform(lambda x: x.shift(2).rolling(window=5).sum())  
    data_monthly['mom36m'] = group.transform(lambda x: x.shift(14).rolling(window=24).sum())  

    return data_monthly

def calculate_beta(stock_returns, market_returns):
    # need more data, the requires at least three years of data
    raise NotImplementedError
    stock_returns = pd.read_csv(os.path.join(p.dataPath, "return_daily.csv"), index_col=False, parse_dates=[0])
    market_returns = pd.read_csv(os.path.join(p.dataPath, "000905_return_daily.csv"), index_col=False, parse_dates=[0])
    stock_returns = stock_returns.rename(columns={stock_returns.columns[0]: 'date'})
    market_returns = market_returns.rename(columns={market_returns.columns[0]: 'date'})

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