import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, Baseline

import parameters as p

class BaseModel:
    def __init__(self):
        self.model = None

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        else:
            raise NotImplementedError
        
class OLSModel(BaseModel):
    def __init__(self, params=p.ols_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = LinearRegression(**params)

class LASSOModel(BaseModel):
    def __init__(self, params=p.lasso_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = Lasso(**params)

class ElasticNetModel(BaseModel):
    def __init__(self, params=p.elasticnet_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = ElasticNet(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

class GBRTModel(BaseModel):
    def __init__(self, params=p.gbrt_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = GradientBoostingRegressor(**params)
    def fit(self, X, y):
        self.model.fit(X, y)

class RFModel(BaseModel):
    def __init__(self, params=p.rf_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = RandomForestRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

class XGBoostModel(BaseModel):
    def __init__(self, params=p.xgboost_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

class NNModel(BaseModel):
    def __init__(self, params=p.nn_params):
        super().__init__()
        if params is None:
            params = {}
        
        input_dim = params.get("input_dim", 1)
        architecture = params.get("architecture", [64, 64])
        output_dim = params.get("output_dim", 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers = []
        for i, units in enumerate(architecture):
            if i == 0:
                layers.append(nn.Linear(input_dim, units))
            else:
                layers.append(nn.Linear(architecture[i-1], units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(architecture[-1], output_dim))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def fit(self, X, y, epochs=10, batch_size=32):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        self.model.train()
        for _ in range(epochs):
            permutation = torch.randperm(X.size()[0])
            for i in range(0, X.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X[indices], y[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.cpu().numpy()
    
def calculate_r2_oos(y_hat, y):
    assert len(y_hat) == len(y)
    n = len(y_hat)
    mean_hat = sum(y_hat) / n
    SSR = sum((y - mean_hat)^2)
    SST = sum(y)
    res = 1 - SSR/SST
    return res


def split_train_val(data_input:pd.DataFrame,
                    target_col:str='alpha', time_col:str='days_from_start',
                    group_cols:list=['Ticker', 'sector'],\
                    
                    time_varying_known_reals:list=["date", 'days_from_start',],
                                                #    'day_of_month', 'day_of_week',
                                                #    'month'],
                    min_prediction_length:int=p.offset, 
                    max_prediction_length:int=p.offset,
                    min_encoder_length:int=p.offset//2,
                    max_encoder_length:int = p.history_length,
                    batch_size:int=64):
    '''
    Initialize the training and validation sets.
    All the parameters are set in the daily_parameters.py file.
    Only one day is used as the validation set.
    NOTE: the target, the group columns, and the time varying known reals are hard coded here.
    '''
    # split the data into training and validation sets
    # max_encoder_length = data_input[time_col].nunique()
    training_cutoff = data_input[time_col].max() - 1

    cols_to_remove = [target_col] + group_cols + [time_varying_known_reals]
    variable_list = [col for col in data_input.columns if col not in cols_to_remove]

    training = TimeSeriesDataSet(
        data_input[lambda x: x[time_col] <= training_cutoff], # type: ignore
        time_idx=time_col,
        target=target_col,
        group_ids=[group_cols[0]],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=group_cols,
        time_varying_known_reals=time_varying_known_reals[:2], # REVIEW: change back to full list 
        time_varying_unknown_reals=[target_col] + variable_list,
        lags={target_col:[1, 5, 25, 50, 75, 252]},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, data_input, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=-1)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=-1)

    return training, validation, train_dataloader, val_dataloader
