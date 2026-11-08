import optuna
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, Metric
from pathlib import Path

# Check available devices
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(device)

# Load Dataset
from sklearn.datasets import fetch_covtype
data = fetch_covtype(data_home=Path.cwd().joinpath('data'), download_if_missing=True, return_X_y=True)
X_train_val, X_test, y_train_val, y_test = train_test_split(*data, test_size=0.05, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.05, random_state=42)

X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
X_val, y_val = torch.FloatTensor(X_val), torch.LongTensor(y_val)
X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)

means = X_train.mean(dim=0, keepdim=True)
stds = X_train.std(dim=0, keepdim=True)

X_train = (X_train - means) / stds
X_val = (X_val - means) / stds
X_test = (X_test - means) / stds

train_data = TensorDataset(X_train, y_train)
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=262144, shuffle=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=262144)

test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=262144)

n_features = data[0].shape[1]
n_classes = data[1].max()

# Define Dense module
class Dense(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
    def forward(self, X):
        return self.model(X)

def evaluate_tm(model, loader, metric_fn: Metric, device_='mps'):
    model.eval()
    metric_fn.reset()
    with torch.no_grad():
        for X_mb, y_mb in loader:
            X_mb, y_mb = X_mb.to(device_), y_mb.to(device_)
            y_pred = model(X_mb)
            metric_fn.update(y_pred, y_mb)
    return metric_fn.compute()

# Hyperparameter tuning
def objective(trial, epochs, accuracy):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e1, log=True)
    model = nn.Sequential(
        Dense(n_features, 400),
        Dense(400, 300),
        Dense(300, 200),
        Dense(200, 100),
        nn.Linear(100, n_classes),
    ).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    val_score = 0.0
    for epoch in range(epochs):
        model.train()
        for X_mb, y_mb in train_loader:
            X_mb, y_mb = X_mb.to(device), y_mb.to(device)
            y_out = model(X_mb)
            loss_ = loss(y_out, y_mb)
            loss_.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_score = evaluate_tm(model, val_loader, accuracy)
        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_score

sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
study = optuna.create_study(study_name='cover-type-classifier-400-300-200-100',
                            direction="maximize",
                            sampler=sampler,
                            storage=f'sqlite:///{Path.cwd().parent.joinpath('models/optuna-storage/cover-type-clf-tuning_dense_400-300-200-100.db')}',
                            load_if_exists=True)
study.optimize(lambda trial: objective(trial, epochs=30, accuracy=Accuracy(task='multiclass', num_classes=int(n_classes)).to(device)), n_trials=30)