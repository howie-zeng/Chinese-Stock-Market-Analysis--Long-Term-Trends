import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import os
import pandas as pd
from tqdm import tqdm
import datetime
import parameters as p

# need volumn traded
def read_data(files):
    merged_df = None
    for file in tqdm(files, desc="Reading files"):
        try:
            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(p.dataPath, file)
            df_temp = pd.read_csv(file_path, index_col=False, parse_dates=[0])
            df_temp = df_temp.rename(columns={df_temp.columns[0]: 'date'})
            df_temp = df_temp.melt(id_vars=['date'], var_name=p.stockID, value_name=file_name)
            if merged_df is None:
                merged_df = df_temp
            else:
                merged_df = pd.merge(merged_df, df_temp, on = ["date", p.stockID], how="left")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            break
    if not merged_df.empty:
        column_name = "return_daily" if "return_daily" in merged_df.columns else "return_monthly"
        data_path = p.merged_data_daily if "return_daily" in merged_df.columns else p.merged_data_monthly
        merged_df = merged_df.dropna(subset=[column_name])
        merged_df.set_index('date', inplace=True)
        merged_df = merged_df.groupby(p.stockID, group_keys=False).apply(lambda x: x.ffill())
        merged_df.to_csv(data_path, index=True) 
    return merged_df

def get_market_data(data):

    df_sector = pd.read_csv(os.path.join(p.dataPath, '000905.csv'), index_col=False, parse_dates=[0])
    df_sector = df_sector.rename(columns={df_sector.columns[0]: 'date', df_sector.columns[1]: '000905_close'})

    df_temp = pd.read_csv(os.path.join(p.dataPath, '000905_return_daily.csv'), index_col=False, parse_dates=[0])
    df_temp = df_temp.rename(columns={df_temp.columns[0]: 'date', df_temp.columns[1]: '000905_return_daily'})

    df_sector = pd.merge(df_sector, df_temp, on='date', how='left')

    df_temp = pd.read_csv(os.path.join(p.dataPath, '000905_return_monthly.csv'), index_col=False, parse_dates=[0])
    df_temp = df_temp.rename(columns={df_temp.columns[0]: 'date', df_temp.columns[1]: '000905_return_monthly'})
    df_temp['month'] = df_temp['date'].dt.to_period('m')
    df_sector['month'] = df_sector['date'].dt.to_period('m')

    df_sector =pd.merge(df_sector, df_temp[['month', '000905_return_monthly']], on='month', how='left')
    df_sector.drop('month', inplace=True, axis=1)

    data = pd.merge(data, df_sector, on='date', how='left')
    
    return data


def data_loading(daily_files, monthly_files, daily_file_path=p.merged_data_daily, monthly_file_path=p.merged_data_monthly):
    dtype = {p.stockID: str}
    # index_col = 'date'
    if os.path.exists(daily_file_path):
        print('Loading daily data from existing file.')
        merged_data_daily = pd.read_csv(p.merged_data_daily, index_col=False, dtype=dtype, parse_dates=[0])
    else:
        print('Daily data file not found. Processing daily files...')
        merged_data_daily = read_data(daily_files)
    if os.path.exists(monthly_file_path):
        print('Loading monthly data from existing file.')
        merged_data_monthly = pd.read_csv(p.merged_data_monthly, index_col=False, dtype=dtype, parse_dates=[0])
    else:
        print('Monthly data file not found. Processing monthly files...')
        merged_data_monthly = read_data(monthly_files)

    return merged_data_daily, merged_data_monthly

def check_datetime(input):
    if not isinstance(input, pd.DatetimeIndex):
        input = pd.to_datetime(input)
    return input


def merge_daily_and_monthly_data(merged_data_daily, merged_data_monthly):
    merged_data_daily.index = check_datetime(merged_data_daily.index)
    merged_data_monthly.index = check_datetime(merged_data_monthly.index)
    merged_data_daily = merged_data_daily.groupby(p.stockID).resample("M").last().reset_index(level=0, drop=True)
    data_monthly = pd.merge(merged_data_daily, merged_data_monthly, on=['date', p.stockID], how='left')
    data_monthly = data_monthly.sort_index()
    # may want to drop some daily columns, but I am keeping it for know
    return data_monthly

def handle_crosssectional_na(data, column_missing_threshold=0.6):
    column_missing_percentage = data.isna().mean()
    excluded_columns = column_missing_percentage[column_missing_percentage > column_missing_threshold].index
    if len(excluded_columns) > 0:
        print(f"Excluding columns with missing values exceeding {column_missing_threshold * 100}%: {', '.join(excluded_columns)}")
        print("\n")

    columns_to_exclude = ['Ticker', 'close', 'close_adj']
    fillna_columns = [col for col in data.columns if col not in columns_to_exclude]

    row_missing_percentage = data[fillna_columns].isna().mean(axis=1)
    row_missing_threshold = row_missing_percentage[row_missing_percentage != 0].mean() + 1.96 * 2 * row_missing_percentage[row_missing_percentage != 0].std()
    data_filtered = data.loc[row_missing_percentage <= row_missing_threshold]
    num_dropped_rows = (row_missing_percentage > row_missing_threshold).sum()
    print(f"Dropped {num_dropped_rows} rows with more than {row_missing_threshold*100}% missing values.")
    print("\n")

    data_filtered = data_filtered.drop(columns=excluded_columns)
    median_by_date = data_filtered.groupby(data_filtered.index.date)[fillna_columns].transform('median')
    data_filtered[fillna_columns] = data_filtered[fillna_columns].fillna(median_by_date)

    missing_statistics_after = data_filtered[fillna_columns].isna().mean()
    change_in_missing_statistics = (column_missing_percentage.loc[fillna_columns] - missing_statistics_after) * 100
    print("Increase in missing statistics for each column:")
    for col, change in change_in_missing_statistics.items():
        print(f"{col}: {change:.2f}%")

    
    data_filtered['beta_000905']
    return data_filtered

def filter_by_market_capitalization(data, threshold = 0.3, variable_name = 'current_assets'):
    mean_capitalization = data.groupby('Ticker').agg({variable_name: np.mean})
    mean_capitalization['rank'] = mean_capitalization[variable_name].rank(pct=True)

    top_70 = mean_capitalization[mean_capitalization['rank'] >  threshold].index
    bottom_30 = mean_capitalization[mean_capitalization['rank'] <=  threshold].index
    
    return top_70, bottom_30

def save_preprocessed_data(data, name, data_path=p.cleanedDataPath):
    data.to_csv(os.path.join(data_path, f"{name}.csv"), index=True) 
    print(f"File {name} saved.")




