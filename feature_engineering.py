import pandas as pd
import numpy as np
import parameters as p
import statsmodels.api as sm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_momentum(data_monthly, col_name = 'return_monthly'):
    data_monthly.sort_index(inplace=True)
    def calculate_group_momentum(group):
        shifted_returns = group[col_name].shift(1)
        group[f'mom1m_{col_name}'] = shifted_returns
        group[f'mom6m_{col_name}'] = shifted_returns.rolling(window=6).apply(lambda x: (1 + x).prod() - 1, raw=True)
        group[f'mom12m_{col_name}'] = shifted_returns.rolling(window=12).apply(lambda x: (1 + x).prod() - 1, raw=True)
        group[f'mom24m_{col_name}'] = shifted_returns.rolling(window=24, min_periods=12).apply(lambda x: (1 + x).prod() - 1)
        group[f'mom36m_{col_name}'] = shifted_returns.rolling(window=36, min_periods=12).apply(lambda x: (1 + x).prod() - 1)
        return group
    data_monthly = data_monthly.groupby(p.stockID).apply(calculate_group_momentum)
    return data_monthly.reset_index(level=0, drop=True)


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
    })

    combined_weekly_returns.dropna(inplace=True)
    
    results = []
    if len(combined_weekly_returns) >= 52:
        X = sm.add_constant(combined_weekly_returns[['market_returns']])
        y = combined_weekly_returns['stock_returns']
        
        for i in range(52, len(y)):
            y_temp = y.iloc[max(i-156, 0):i]
            X_temp = X.iloc[max(i-156, 0):i]
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
    merged_data['betasq'] = merged_data['beta']**2
    return merged_data

def calculate_smas(data, windows=[5, 10, 20, 50, 100, 200], close="close_adj"):
    for window in windows:
        data[f'SMA_{window}'] = data[close].rolling(window=window, min_periods=1).mean()
    return data

def calculate_macd(data, close='close_adj', fast_period=12, slow_period=26, signal_period=9):
    data['EMA_fast'] = data[close].ewm(span=fast_period, adjust=False).mean()
    data['EMA_slow'] = data[close].ewm(span=slow_period, adjust=False).mean()
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    return data.drop(['EMA_fast', 'EMA_slow'], axis=1)

def calculate_bollinger_bands(data, close='close_adj', window=20, num_of_std=2):
    data['SMA'] = data[close].rolling(window=window, min_periods=1).mean()
    data['STD'] = data[close].rolling(window=window, min_periods=1).std()
    data['Upper_Band'] = data['SMA'] + (data['STD'] * num_of_std)
    data['Lower_Band'] = data['SMA'] - (data['STD'] * num_of_std)
    return data.drop(['STD'], axis=1)

def calculate_rsi(data, close='close_adj', period=14):
    delta = data[close].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_roc(data, close='close_adj', period=10):
    data['ROC'] = ((data[close] - data[close].shift(period)) / data[close].shift(period)) * 100
    return data

def calculate_historical_volatility(data, return_col='return_daily', windows=[252, 120, 60]):
    for window in windows:
        daily_vol = data[return_col].rolling(window=window).std()
        data[f"volatility_{window}"] = daily_vol * np.sqrt(252)  # annualizing
    return data

def calculate_technical_indicators(data_daily):
    functions = [calculate_macd, calculate_bollinger_bands, calculate_rsi, 
                 calculate_roc, calculate_historical_volatility, calculate_smas
                ]
    for f in functions:
        data_daily = f(data_daily)
    return data_daily

def process_stock_daily_data(stock_data):
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data.set_index('date', inplace=True)
    result = calculate_technical_indicators(stock_data)
    result.reset_index(inplace=True)
    return result

def calculate_ema(data, column='zero_trade_days', spans=[6, 12, 24, 36]):
    for span in spans:
        ema_col_name = f'{column}_EMA_{span}'
        data[ema_col_name] = data.groupby(p.stockID)[column].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    return data

def feature_construction(data_daily, data_monthly): 
    # daily
    print('Calculating Daily Features')
    data_daily['month'] = data_daily['date'].dt.to_period('m')
    maxret = data_daily.groupby([p.stockID, 'month'])['return_daily'].max().reset_index().rename(columns={'return_daily': 'maxret'})
    data_daily = pd.merge(data_daily, maxret, on=[p.stockID, 'month'], how='left')

    # indicators
    print('Calculating Technical Indicators')
    grouped_daily = data_daily.groupby(p.stockID)
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_stock_daily_data, [group.copy() for name, group in grouped_daily])
    data_daily = pd.concat(results)
    # volatility = data_daily.groupby([p.stockID, 'month'])['return_daily'].std().rename(columns={'return_daily': 'volatility'})
    # data_daily = pd.merge(data_daily, volatility, left_on=[p.stockID, 'month'], right_on=[p.stockID, 'month'], how='left')

    # monthly
    print('Calculating Monthly Features')
    chmom_6m = data_monthly.groupby(p.stockID)['return_monthly'].rolling(window=6).apply(lambda x: (1+x).prod() - 1, raw=True).shift(1).reset_index(level=0, drop=True)
    chmom_12m = data_monthly.groupby(p.stockID)['return_monthly'].rolling(window=6).apply(lambda x: (1+x).prod() - 1, raw=True).shift(7).reset_index(level=0, drop=True)
    data_monthly['chmom'] = chmom_6m - chmom_12m
    data_monthly = calculate_momentum(data_monthly, col_name = 'return_monthly')
    data_monthly = calculate_momentum(data_monthly, col_name = '000905_return_monthly')
    data_monthly = calculate_ema(data_monthly, column='zero_trade_days')

    data_monthly.loc[:, 'excess_return'] = data_monthly.loc[:, 'return_monthly'] - 0
    data_monthly['y'] = data_monthly[['excess_return', p.stockID]].groupby(p.stockID).shift(-1)

    data_daily.drop('month', axis=1, inplace=True)
    data_daily.set_index('date', inplace=True)
    data_monthly.set_index('date', inplace=True)
    return data_daily, data_monthly

def calculate_pct_from_close_adj(price, close):
    pct = (price - close)/close
    return pct

def transform_features(data, scaler_name = "standard"):
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler name: {scaler_name}")
    
    data['Upper_Band'] = calculate_pct_from_close_adj(data['Upper_Band'], data['close_adj'])
    data['Lower_Band'] = calculate_pct_from_close_adj(data['Lower_Band'], data['close_adj'])

    def scale_group(group_df):
        scaled_values = scaler.fit_transform(group_df[cols_to_transform])
        group_df[cols_to_transform] = scaled_values
        return group_df

    cols_to_transform = ['SMA',
                         'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
                         '000905_close', 'close_adj', '000905_close'
                        ]
    data_scaled = data.groupby(p.stockID).apply(scale_group)
    data_scaled.reset_index(level=0, inplace=True, drop=True)

    return data_scaled
                                                                 
                                                                 
