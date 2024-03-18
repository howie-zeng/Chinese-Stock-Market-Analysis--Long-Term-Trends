import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


offset = 252

def data_loading(data):
    data.drop(data.index[:offset], inplace=True)
    data.index = pd.to_datetime(data.index)
    data['sector'] = data['sector'].fillna("None")
    data['alpha'] = data['alpha'].fillna(method='bfill')
    data['beta_market'] = data['beta_market'].fillna(method='bfill')
    data['beta_sector'] = data['beta_sector'].fillna(method='bfill')
    
    data = data.groupby('Ticker').apply(lambda x: x.fillna(method='ffill'))
    data.set_index('date', inplace=True)
    
    return data

def feature_filter(data, percent=0.9):
    data_isnotna = data.groupby("Ticker").count()
    num_tickers_for_features = data_isnotna.apply(lambda x: sum(x > 0), axis=0)
    features_to_use = num_tickers_for_features[num_tickers_for_features > percent * len(data_isnotna)].index
    features_to_use = list(features_to_use) + ["Ticker"]
    data_fs = data[features_to_use]
    data_fs_isnotna = data_fs.groupby("Ticker").count()
    tickers_with_na_features = data_fs_isnotna.apply(lambda x: sum(x == 0), axis=1)
    tickers_to_use = tickers_with_na_features[tickers_with_na_features == 0].index
    return data_fs[data_fs['Ticker'].isin(tickers_to_use)]

def data_preprocessing(data_fs):
    date_col = pd.to_datetime(data_fs.index)
    earliest_time = pd.to_datetime(data_fs.index.min())
    f_dict = {}
    day_cnt = 0
    max_diff = (date_col - earliest_time).days.nunique()
    for day in date_col.unique():
        if day_cnt <= max_diff:
            f_dict[day] = day_cnt
        day_cnt += 1
    data_fs['date'] = date_col
    data_fs['days_from_start'] = [f_dict[d] for d in date_col]
    # data_fs['day_of_month'] = date_col.day # type: ignore
    # data_fs['day_of_week'] = date_col.dayofweek # type: ignore
    # data_fs['month'] = date_col.month # type: ignore
    data_fs.sort_index(inplace=True)

    data_fs = data_fs.reset_index(drop=True)
    return data_fs

def fillnas_and_convert(data, stockID="Ticker", dataOffset='W', fill_methods=("ffill", "bfill"), missing_threshold=0.6):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    for method in fill_methods:
        data = data.groupby(stockID, group_keys=False).apply(lambda x: x.fillna(method=method))
    data = data.groupby(stockID).resample(dataOffset).last().reset_index(level=0, drop=True)
    missing_percentage = data.isna().mean()
    selected_columns = missing_percentage[missing_percentage <= missing_threshold].index
    data = data[selected_columns]
    median_by_date = data.groupby(data.index.date).transform('median')
    data = data.fillna(median_by_date)
    return data

def filter_by_market_capitalization(data, threshold = 0.3, variable_name = 'current_assets'):
    mean_capitalization = data.groupby('Ticker').agg({variable_name: np.mean})
    mean_capitalization['rank'] = mean_capitalization[variable_name].rank(pct=True)

    top_70 = mean_capitalization[mean_capitalization['rank'] >  threshold].index
    bottom_30 = mean_capitalization[mean_capitalization['rank'] <=  threshold].index
    
    return top_70, bottom_30

