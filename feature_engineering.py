import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, xgb  
import xgboost as xgb


import torch
import torch.nn as nn
import torch.optim as optim

class BaseModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        else:
            raise NotImplementedError
        
class OLSModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
    
class PLSModel(BaseModel):
    def __init__(self, n_components=2):
        super().__init__()
        self.model = PLSRegression(n_components=n_components)

class LASSOModel(BaseModel):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.model = Lasso(alpha=alpha)

class ElasticNetModel(BaseModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        super().__init__()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

class GBRTModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        super().__init__()
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

class RFModel(BaseModel):
    def __init__(self, n_estimators=100):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=n_estimators)

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

class NNModel(BaseModel):
    def __init__(self, input_dim, architecture=[64, 64], output_dim=1):
        super().__init__()
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