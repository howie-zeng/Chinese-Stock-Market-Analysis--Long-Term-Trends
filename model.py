from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy
import optuna 
from tqdm import tqdm


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
        self.model = HuberRegressor(**params)

class OLS3Model(BaseModel):
    def __init__(self, params=p.ols3_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = HuberRegressor(**params)

class PLSModel(BaseModel):
    def __init__(self, params=p.pls_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = PLSRegression(**params)

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
        self.model = RandomForestRegressor(**params, n_jobs=-1)

class XGBoostModel(BaseModel):
    def __init__(self, params=p.xgboost_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = xgb.XGBRegressor(**params, n_jobs=-1)

class VASAModel(BaseModel):
    def __init__(self, params=p.vasa_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = None

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


    def fit(self, X, y, num_epochs=50, batch_size=32):
        # X = self.preprocess_nas(X)
        if isinstance(X, pd.DataFrame):
            X_numpy = X.values.astype(np.float32)
        else:
            X_numpy = X.astype(np.float32)
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        patience_count = 0

        progress_bar = tqdm(range(num_epochs), desc="Training")
        for epoch in progress_bar:
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad() 
                output = self.model(batch_X)  
                loss = self.criterion(output, batch_y) 
                loss.backward()
                self.optimizer.step()  

                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(dataloader.dataset)
            progress_bar.set_postfix({"loss": epoch_loss})
    
    def predict(self, X):
        # X = self.preprocess_nas(X)
        if isinstance(X, pd.DataFrame):
            X_numpy = X.values.astype(np.float32)
        else:
            X_numpy = X.astype(np.float32)
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()
    
def split_dates(data_index):
    if not isinstance(data_index, pd.DatetimeIndex):
        data_index = pd.to_datetime(data_index)
    data_index = data_index.sort_values()
    total_size = len(data_index)
    train_size = int(total_size * 0.6)
    val_test_size = (total_size - train_size) // 2

    train_dates = [data_index[0].strftime('%Y-%m-%d'), data_index[train_size - 1].strftime('%Y-%m-%d')]
    val_dates = [data_index[train_size].strftime('%Y-%m-%d'), data_index[train_size + val_test_size - 1].strftime('%Y-%m-%d')]
    test_dates = [data_index[train_size + val_test_size].strftime('%Y-%m-%d'), data_index[-1].strftime('%Y-%m-%d')]

    return train_dates, val_dates, test_dates
      
def split_train_val_test(data, y="return_monthly", colsToDrop = []):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data['y'] = data[y].shift()
    for col in [p.stockID] + colsToDrop:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    data = data.dropna(subset=['y'])
    training_sample, validation_sample, testing_sample = split_dates(data.index)
    training_start, training_end = pd.to_datetime(training_sample)
    validation_start, validation_end = pd.to_datetime(validation_sample)
    testing_start, testing_end = pd.to_datetime(testing_sample)

    training_set = data[(data.index >= training_start) & (data.index <= training_end)]
    validation_set = data[(data.index >= validation_start) & (data.index <= validation_end)]
    testing_set = data[(data.index >= testing_start) & (data.index <= testing_end)]

    X_train, y_train = training_set.drop(columns=['y']), training_set['y']
    X_val, y_val = validation_set.drop(columns=['y']), validation_set['y']
    X_test, y_test = testing_set.drop(columns=['y']), testing_set['y']

    return X_train, y_train, X_val, y_val, X_test, y_test

def train(X, y, model):
    if model.name == "OLS3Model":
        X = X[p.ols3_predictors]
    scaler = StandardScaler()
    model.fit(scaler.fit_transform(X), y)
    return model, scaler

def validation(X, model_fitted, scaler):
    if model_fitted.name == "OLS3Model":
        X = X[p.ols3_predictors] 
    predictions = model_fitted.predict(scaler.transform(X))
    return predictions






