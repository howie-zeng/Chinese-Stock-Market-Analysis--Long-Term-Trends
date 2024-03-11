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
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import copy

import parameters as p

class BaseModel:
    def __init__(self):
        self.model = None

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        else:
            raise NotImplementedError
            
    def fit(self, X, y):
        return self.model.fit(X, y)
             
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

class GBRTModel(BaseModel):
    def __init__(self, params=p.gbrt_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = GradientBoostingRegressor(**params)

class RFModel(BaseModel):
    def __init__(self, params=p.rf_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = RandomForestRegressor(**params)

class XGBoostModel(BaseModel):
    def __init__(self, params=p.xgboost_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = xgb.XGBRegressor(**params)

class NNModel(BaseModel):
    def __init__(self, params=p.nn_params, input_dim=1, num_layers=1):
        super().__init__()
        if params is None:
            params = {}
        self.input_dim = input_dim
        self.num_layers = num_layers
        architecture = params.get("architecture", [64] * num_layers)
        output_dim = params.get("output_dim", 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers = []
        for i, units in enumerate(architecture):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, units))
            else:
                layers.append(nn.Linear(architecture[i-1], units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(architecture[-1], output_dim))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.name = f"{self.__class__.__name__}_nn{num_layers}"
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
    def preprocess_nas(self, X):
        X_mask = X.isnull().astype(float)
        X_preprocessed = pd.concat([X, X_mask], axis=1)
        return X_preprocessed

    def fit(self, X, Y, num_epochs=5, batch_size=32):
        # X = self.preprocess_nas(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:  
                self.optimizer.zero_grad() 
                output = self.model(batch_X)  
                loss = self.criterion(output, batch_y) 
                loss.backward()
                self.optimizer.step()  

            #     epoch_loss += loss.item() * batch_X.size(0)
            # epoch_loss /= len(dataloader.dataset)
            # print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    def predict(self, X):
        # X = self.preprocess_nas(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()

        

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

def train(data: pd.DataFrame, model, start = p.training_sample[0], end = p.training_sample[1]):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data_training = data.loc[(data.index >= start) & (data.index <= end)]
    data_training = data_training.sort_index()
    stock_ticker = data_training['Ticker'].unique()
    model_dict = {}
    scaler_dict = {}
    for stock in stock_ticker:
        model_clone = copy.deepcopy(model)
        model_name = f"{model.__class__.__name__}_{stock}"

        data_stock = data_training[data_training['Ticker'] == stock]
        X = data_stock.drop(['alpha', 'Ticker', 'sector'],axis=1)
        Y = data_stock['alpha']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        scaler_dict[stock] = scaler

        model_clone.fit(X, Y)
        model_dict[model_name] = model_clone
    return model_dict, scaler_dict

def validation(data: pd.DataFrame, model_dict, scaler_dict, start = p.validation_sample[0], end = p.validation_sample[1]):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data_validation = data.loc[(data.index >= start) & (data.index <= end)]
    data_validation = data_validation.sort_index()
    stock_ticker = data_validation['Ticker'].unique()
    model_type = list(model_dict.keys())[0].split("_",1)[0]
    res_validation = []
    for stock in stock_ticker:
        data_stock = data_validation[data_validation['Ticker'] == stock]
        X = data_stock.drop(['alpha', 'Ticker', 'sector'],axis=1)
        Y = data_stock['alpha']

        scaler = scaler_dict[stock]
        X = scaler.transform(X)

        model_name = f"{model_type}_{stock}"
        if model_name in model_dict:
            model = model_dict[model_name]
            predictions = model.predict(X)
            for true_value, prediction in zip(Y, predictions):
                res_validation.append((true_value, prediction, stock))
    results_df = pd.DataFrame(res_validation, columns=['Y', 'prediction', 'Ticker'])
    return results_df

