import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim

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
