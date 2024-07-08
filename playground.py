import torch
import pandas as pd
import numpy as np
from copy import copy
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup
ensemble = 32
batch_size = 64
n_epochs = 0
loss_fn = 'mse'  # 'trade_loss'
context = 32

batch_size = batch_size * ensemble

# prep data
path = 'stock_data/portfolio_custom129_2002_2023_normrange.csv'
data = pd.read_csv(path, index_col=0)
data = np.where(data.values == 0, -1*np.ones_like(data.values), data.values)
data = torch.tensor(data, dtype=torch.float32)
train_data = data[:, 0].unsqueeze(1)[:-11]
eval_data = data[:, 0].unsqueeze(1)[-11:]
# test_data = data[:, 10].unsqueeze(1)[-1000:]
test_data = data[-365:, :-2]   # without cryptos right now
# kick out all 


class DatasetRobobobo(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        xs = self.data[index:index+self.seq_len+1].clone()
        for j in range(xs.shape[1]):
            mask = torch.where(xs[:, j] == -1, torch.ones_like(xs[:, j], dtype=torch.bool), torch.zeros_like(xs[:, j], dtype=torch.bool))
            if torch.any(mask):
                xs[:, j] = torch.zeros_like(xs[:, j], dtype=torch.float32) - 1
            else:
                xs_min = xs[:-1, j].min()
                xs_max = xs[:-1, j].max()
                xs[:, j] = (xs[:, j] - xs_min) / (xs_max - xs_min)

                xs[:, j] = xs[:, j] - xs[:, j][0]
                
        return xs[:-1], xs[-1]
    
train_data_loader = torch.utils.data.DataLoader(DatasetRobobobo(train_data, 10), batch_size=None, shuffle=False)
train_data_prepped = []
for sample in train_data_loader:
    train_data_prepped.append(sample)

eval_data_loader = torch.utils.data.DataLoader(DatasetRobobobo(eval_data, 10), batch_size=None, shuffle=False)
eval_data_prepped = []
for sample in eval_data_loader:
    eval_data_prepped.append(sample)

class TradeLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, order: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return - torch.mean(order * y)

class TradeNet(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.input_dim = 1#data.shape[-1]
        self.hidden_dim = 64
        self.dropout = torch.nn.Dropout(0.25)
        self.linear_in = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.linear_out = torch.nn.Linear(self.hidden_dim, self.input_dim)
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x: torch.Tensor, vote=None) -> torch.Tensor:
        # mask = torch.where(x == -1, torch.zeros_like(x, dtype=torch.bool), torch.ones_like(x, dtype=torch.bool))
        x = self.dropout(self.linear_in(x))
        x = self.dropout(self.lstm(x)[0][:, -1])
        x = self.linear_out(x)
        x = self.tanh(x)
        return x
    
    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, index: int) -> torch.nn.Module:
        return self

class EnsembleNet:
    
    def __init__(self, n_models: int) -> None:
        self.models = [TradeNet() for _ in range(n_models)]
        self.loss_fn = TradeLoss()
        
    def __len__(self) -> int:
        return len(self.models)
    
    def __getitem__(self, index: int) -> TradeNet:
        return self.models[index]
    
    def __call__(self, x: List[torch.Tensor], vote: bool=False) -> torch.Tensor:
        if not isinstance(x, List):
            x = [x for _ in range(len(self.models))]
            
        y_preds = []
        for i, model in enumerate(self.models):
            y_preds.append(model(x[i]))
        if vote:
            return torch.stack(y_preds).median(dim=0)[0]
        else:
            return y_preds
    
    def train(self):
        for model in self.models:
            model.train()
            
    def eval(self):
        for model in self.models:
            model.eval()
            
    def state_dict(self) -> List[dict]:
        return [model.state_dict() for model in self.models]
    
    def load_state_dict(self, state_dict: List[dict]):
        for i, model in enumerate(self.models):
            model.load_state_dict(state_dict[i])
            
    def to(self, device: torch.device):
        for model in self.models:
            model.to(device)
        return self
        
if ensemble > 1:
    model = EnsembleNet(ensemble).to(device)
    optimizer = [torch.optim.Adam(model[i].parameters(), lr=0.001) for i in range(len(model))]
else:
    model = TradeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if loss_fn == 'mse':
    loss_fn = torch.nn.functional.mse_loss
else:
    loss_fn = TradeLoss()

num_batches = batch_size // len(model)
if num_batches == 0:
    raise ValueError('Batch size too small for ensemble')

for e in range(n_epochs):
    model.train()
    n_samples = 0
    loss_batch = 0
    # shuffle train data along batch dimension
    np.random.shuffle(train_data_prepped)
    for index in range(0, len(train_data_prepped), batch_size):
        if index + batch_size > len(train_data_prepped):
            batch_size_now = len(train_data_prepped) - index
        else:
            batch_size_now = batch_size
        batch = train_data_prepped[index:index+batch_size_now]
        x = torch.stack([b[0] for b in batch]).to(device)
        y = torch.stack([b[1] for b in batch]).to(device)
        if ensemble > 1:
            # split into ensemble batches
            x = [x[i*num_batches:(i+1)*num_batches] for i in range(ensemble)]
            y = [y[i*num_batches:(i+1)*num_batches] for i in range(ensemble)]
        if x[-1].shape[0] > 0:
            y_pred = model(x)
            # loss = torch.nn.functional.mse_loss(y_pred, y)
            if ensemble > 1:
                losses = []
                for i in range(len(model)):
                    optimizer[i].zero_grad()
                    losses.append(loss_fn(y_pred[i], y[i]))
                    losses[i].backward()
                    optimizer[i].step()
                loss = torch.stack(losses).mean()
            else:
                optimizer.zero_grad()
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
            
            loss_batch += loss.item()
            n_samples += 1
    msg = f'Epoch {e} - Loss: {loss_batch/n_samples:.6f}'
    
    if e % 10 == 0:
        # test model
        model.eval()
        with torch.no_grad():
            x, y = eval_data_prepped[0]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            y_pred = model(x, vote=True)
            # loss = torch.nn.functional.mse_loss(y_pred, y.to(device))
            loss = loss_fn(y_pred, y.to(device))
            msg += f' - Test loss: {loss.item():.6f}'
    
    print(msg)
    
    # save model
    torch.save(model.state_dict(), 'model.pt')

# load model
model.load_state_dict(torch.load('model.pt'))

class TradingAlgorithm:
    
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        
    def __call__(self, x: torch.Tensor, vote: bool) -> torch.Tensor:
        # 1. calls the model to get predictions on each stock (features)
        # 2. the best 10% of stocks are bought
        # 3. the rest is sold
        
        # 1. get predictions
        y_preds = torch.zeros(x.shape[-1])
        for i in range(x.shape[-1]):
            y_preds[i] = self.model(x[:, :, i].unsqueeze(-1), vote=True)
        
        # 2. buy best 10% of stocks TODO: which can be afforded
        index_buy = torch.argsort(y_preds, descending=True)[:int(0.1*len(y_preds))]
        
        # 3. sell rest
        index_sell = torch.argsort(y_preds, descending=True)[int(0.1*len(y_preds)):]
        
        return index_buy, index_sell

# test model on test data
model.eval()
robobobo = TradingAlgorithm(model)
cash = 10000
n_stocks = torch.zeros(test_data.shape[-1])
equity = []
equity_comp = []
n_stocks_comp = cash / test_data.shape[-1] / test_data[9]
for i in range(len(test_data)-11):
    print(f'Progress: {i}/{len(test_data)-11}', end='\r')
    # get data characteristics
    current_price = test_data[i+9]
    next_price = test_data[i+10]
    
    # prepare data
    x = test_data[i:i+10].unsqueeze(0)
    x_min = x.min(dim=1)[0]
    x_max = x.max(dim=1)[0]
    x = (x - x_min.unsqueeze(1)) / (x_max.unsqueeze(1) - x_min.unsqueeze(1))
    x = x - x[:, 0]
    x = x.to(device)
    
    # get trade orders
    index_buy, index_sell = robobobo(x, True)
    # sell
    cash += torch.sum(n_stocks[index_sell] * current_price[index_sell])
    n_stocks[index_sell] = 0
    # buy
    cash_weighted = torch.ones(len(index_buy)) * (cash / len(index_buy))
    new_stocks = cash_weighted // current_price[index_buy]
    n_stocks[index_buy] += new_stocks
    cash -= torch.sum(new_stocks * current_price[index_buy])
    
    # add new equity
    equity_comp.append(torch.sum(n_stocks_comp * next_price))
    equity.append(copy(cash) + torch.sum(copy(n_stocks) * copy(next_price)))

equity = np.array(equity)
equity_comp = np.array(equity_comp)

import matplotlib.pyplot as plt

plt.plot(equity_comp)
plt.plot(equity)
plt.show()