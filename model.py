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
import evaluation as e


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
        self.model = HuberRegressor(**params, max_iter=5000)

class OLS3Model(BaseModel):
    def __init__(self, params=p.ols3_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = HuberRegressor(**params, max_iter=5000)

class PLSModel(BaseModel):
    def __init__(self, params=p.pls_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = PLSRegression(**params, max_iter=5000)

class LASSOModel(BaseModel):
    def __init__(self, params=p.lasso_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = Lasso(**params, max_iter=5000)

class ElasticNetModel(BaseModel):
    def __init__(self, params=p.elasticnet_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = ElasticNet(**params, max_iter=5000)

class GBRTModel(BaseModel):
    def __init__(self, params=p.gbrt_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = GradientBoostingRegressor(**params, random_state=p.RandomState)

class RFModel(BaseModel):
    def __init__(self, params=p.rf_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = RandomForestRegressor(**params, n_jobs=-1, random_state=p.RandomState)

class XGBoostModel(BaseModel):
    def __init__(self, params=p.xgboost_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = xgb.XGBRegressor(**params, n_jobs=-1, random_state=p.RandomState)

class VASAModel(BaseModel):
    def __init__(self, params=p.vasa_params):
        super().__init__()
        if params is None:
            params = {}
        self.model = None

class NNModel(BaseModel):
    def __init__(self, params=p.nn_params, input_dim=1, num_layers=1):
        super().__init__()
        if params is None or len(params) < 1:
            params = {
                'architecture': [32, 16, 8, 4, 2][:num_layers], 
                'loss': 'mse',                   
                'lambda': 0.0001,                 
                'learning_rate': 0.001,       
                'batch_size': 64,               
                'output_dim': 1                 
            }
        self.input_dim = input_dim
        self.num_layers = num_layers
        architecture = params['architecture']
        output_dim = params['output_dim']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers =  self._build_layers(input_dim, architecture, output_dim)
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.name = f"{self.__class__.__name__}_nn{num_layers}"
        self.batch_size = params['batch_size']

        self.criterion = self._select_loss_function(params['loss'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'], weight_decay=params['lambda'])
        
    def _build_layers(self, input_dim, architecture, output_dim):
        layers = []
        for i, units in enumerate(architecture):
            layers.append(nn.Linear(input_dim if i == 0 else architecture[i-1], units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(architecture[-1], output_dim))
        return layers   
    
    def _select_loss_function(self, loss_name):
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'huberloss':
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def preprocess_nas(self, X):
        X_mask = X.isnull().astype(float)
        X_preprocessed = pd.concat([X, X_mask], axis=1)
        return X_preprocessed

    def fit(self, X, y, X_val=None, y_val=None, num_epochs=100, batch_size=64, patience=5):
        if isinstance(X, pd.DataFrame):
            X_numpy = X.values.astype(np.float32)
        else:
            X_numpy = X.astype(np.float32)
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values.astype(np.float32), dtype=torch.float32).to(self.device)

        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_numpy = X_val.values.astype(np.float32)
            else:
                X_val_numpy = X_val.astype(np.float32)
            X_val_tensor = torch.tensor(X_val_numpy, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val.values.astype(np.float32), dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        progress_bar = tqdm(range(num_epochs), desc="Training")
        
        for epoch in progress_bar:
            self.model.train()  # Set the model to training mode
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad() 
                output = self.model(batch_X).squeeze()  
                loss = self.criterion(output, batch_y) 
                loss.backward()
                self.optimizer.step()  
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(dataloader.dataset)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                self.model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    val_output = self.model(X_val_tensor).squeeze()
                    val_loss = self.criterion(val_output, y_val_tensor).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0  # Reset counter
                    else:
                        epochs_without_improvement += 1
                    
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break  # Stop training
            
            progress_bar.set_postfix({"Training loss": epoch_loss, "Validation loss": val_loss if X_val is not None and y_val is not None else "N/A"})
        
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
        return predictions.cpu().numpy().flatten()
    
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
      
def split_train_val_test(data, y="excess_return", colsToDrop = []):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
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

def train(X, y, model, X_val=None, y_val=None):
    if model.name == "OLS3Model":
        X = X[p.ols3_predictors]
    scaler = StandardScaler()
    if 'NNModel' in model.name:
        model.fit(scaler.fit_transform(X), y, scaler.transform(X_val), y_val)
    else:
        model.fit(scaler.fit_transform(X), y)
    return model, scaler

def test(X, model_fitted, scaler):
    if model_fitted.name == "OLS3Model":
        X = X[p.ols3_predictors] 
    predictions = model_fitted.predict(scaler.transform(X))
    return predictions

def fit_model(X_train, X_test,  X_val, y_val, y_train, y_test, model_classes):
    model_r_2 = {}
    models_fitted = {}
    model_res = pd.DataFrame()
    model_res.index = y_test.index
    model_res['y'] = y_test
    progress_bar = tqdm(model_classes, desc="Testing")
    for model_class in progress_bar:
        model_name = model_class.name if hasattr(model_class, "name") else model_class.__class__.__name__
        model_class.name = model_name
        print(model_name)
        if 'NNModel' in model_name:
            model_fitted, scaler = train(X_train, y_train, model_class, X_val, y_val)
        else:
            model_fitted, scaler = train(X_train, y_train, model_class)
        predictions = test(X_test, model_fitted, scaler)
        r_2 = e.calculate_r2_oos(y_test.values, predictions)

        models_fitted[model_name] = model_fitted
        model_res[model_name] = predictions
        model_r_2[model_name] = r_2
        progress_bar.set_postfix({"Model Name": model_name, "R_2": r_2})
    for model_name, r_2 in model_r_2.items():
        print(f"{model_name}: {r_2}")
    return models_fitted, model_res, model_r_2






