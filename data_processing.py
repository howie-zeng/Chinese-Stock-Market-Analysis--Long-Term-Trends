import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


offset = 252

def data_loading(data):
    data.drop(data.index[:offset], inplace=True)
    data.index = pd.to_datetime(data.index)
    data['sector'] = data['sector'].fillna("None")
    # Use back fill to fill the missing values for alpha and betas
    data['alpha'] = data['alpha'].fillna(method='bfill')
    data['beta_market'] = data['beta_market'].fillna(method='bfill')
    data['beta_sector'] = data['beta_sector'].fillna(method='bfill')
    
    data = data.groupby('Ticker').apply(lambda x: x.fillna(method='ffill'))
    data.set_index('date', inplace=True)
    
    return data

def feature_filter(data, percent=0.9):
    '''
    filter features with too little tickers
    filter tickers with too little features
    '''
    data_isnotna = data.groupby("Ticker").count()

    # filter features with more than percent% tickers with non-na values
    num_tickers_for_features = data_isnotna.apply(lambda x: sum(x > 0), axis=0)
    features_to_use = num_tickers_for_features[num_tickers_for_features > percent * len(data_isnotna)].index
    features_to_use = list(features_to_use) + ["Ticker"]  # append 'ticker' to the list of features

    # filter tickers according to the number of features they have
    data_fs = data[features_to_use]
    data_fs_isnotna = data_fs.groupby("Ticker").count()
    tickers_with_na_features = data_fs_isnotna.apply(lambda x: sum(x == 0), axis=1)
    tickers_to_use = tickers_with_na_features[tickers_with_na_features == 0].index
    return data_fs[data_fs['Ticker'].isin(tickers_to_use)]


def rf_feature_selection(X, Y):
    '''
    Use Random Forest Regressor to select the most important features
    Currently, the function is not used in the workflow.
    '''
    # Then use a random forest regressor to return a list of the most important features
    # Here, we use median importance as the threshold (default)
    rf_selector = SelectFromModel(RandomForestRegressor())
    rf_selector.fit(X,Y)
    
    rf_support = rf_selector.get_support()
    rf_feature = X.loc[:,rf_support].columns.tolist() # type: ignore

    return rf_feature, rf_selector

def data_preprocessing(data_fs):
    '''
    A workflow to preprocess the data.
    Add time features into the data.
    '''
    
    # other preprocessing
    date_col = pd.to_datetime(data_fs.index)
    earliest_time = pd.to_datetime(data_fs.index.min())
    f_dict = {}
    day_cnt = 0
    max_diff = (date_col - earliest_time).days.nunique()
    for day in date_col.unique():
        if day_cnt <= max_diff:
            f_dict[day] = day_cnt
        day_cnt += 1

    # TODO: add quarter (+ last reported quarter)
    data_fs['date'] = date_col
    data_fs['days_from_start'] = [f_dict[d] for d in date_col]
    # data_fs['day_of_month'] = date_col.day # type: ignore
    # data_fs['day_of_week'] = date_col.dayofweek # type: ignore
    # data_fs['month'] = date_col.month # type: ignore
    data_fs.sort_index(inplace=True)

    data_fs = data_fs.reset_index(drop=True)
    return data_fs