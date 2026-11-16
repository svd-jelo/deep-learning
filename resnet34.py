import torch.nn as nn
import torch
import numpy as np
import copy
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
import torchvision.transforms.v2 as T
from pathlib import Path
from functools import partial

device = 'mps'


# Implemention ResNet-34
# Residual unit
class ResidualUnit(nn.Module):
    """
    restrictions:
     - padding is set to "same"
     - only two convolutions per unit
     - only the first convolution can have a stride > 1
     - fixed kernel size
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        DefaultConv2d = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)
        self.deep_stack = nn.Sequential(
            DefaultConv2d(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            DefaultConv2d(in_channels=out_channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
        )

        if stride > 1:
            self.wide_stack = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.wide_stack = nn.Identity()

        self.output = nn.ReLU()

    def forward(self, X):
        Z = self.deep_stack(X) + self.wide_stack(X)
        return self.output(Z)


# ResNet-34 - images must have height and width > 32
# Starting layers
resnet34_layers: list[nn.Module] = [
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
]
# 3 RUs, 64 fmaps
for i in range(3):
    resnet34_layers.append(
        ResidualUnit(64, 64)
    )
# 4 RUs, 128 fmaps
resnet34_layers.append(ResidualUnit(64, 128, 2))
for i in range(3):
    resnet34_layers.append(
        ResidualUnit(128, 128)
    )
# 6 RUs, 256 fmaps
resnet34_layers.append(ResidualUnit(128, 256, 2))
for i in range(5):
    resnet34_layers.append(
        ResidualUnit(256, 256)
    )
# 3 RUs, 512 fmaps
resnet34_layers.append(ResidualUnit(256, 512, 2))
for i in range(2):
    resnet34_layers.append(
        ResidualUnit(512, 512)
    )
# Global Average Pooling
resnet34_layers.append(
    nn.AdaptiveAvgPool2d(output_size=1)
)
# Fully connected 1000
resnet34_layers.extend([
    nn.Flatten(),
    nn.Linear(512, 10)
])

resnet34 = nn.Sequential(*resnet34_layers).to(device)

# Load CIFAR10 dataset
cifar10_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
train_val_data = datasets.CIFAR10(root=Path.cwd().joinpath('data'), download=True, train=True, transform=cifar10_transform)
test_data = datasets.CIFAR10(root=Path.cwd().joinpath('data'), download=True, train=False, transform=cifar10_transform)

# Split train val data
train_data, val_data = random_split(train_val_data, [45000, 5000])

# Define loaders
torch.manual_seed(42)
cifar10_train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
cifar10_val_loader = DataLoader(val_data, batch_size=128)
cifar10_test_loader = DataLoader(test_data, batch_size=128)


# Train ResNet-34 on the CIFAR10 dataset
def he_init_resnet(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight)

torch.manual_seed(42)
resnet34.apply(he_init_resnet)

def train_resnet(model, train_loader, val_loader, optimizer, scheduler, criterion, n_epochs):
    learning_curve = []
    best_val_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(n_epochs):
        # Training loss
        model.train()
        training_loss = 0.0
        for X_mb, y_mb in train_loader:
            X_mb, y_mb = X_mb.to(device), y_mb.to(device)
            y_pred = model(X_mb)
            loss = criterion(y_pred, y_mb)
            training_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        training_loss /= len(train_loader)

        if epoch + 1 in [250, 375]:
            scheduler.step()

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_mb, y_val_mb in val_loader:
                X_val_mb, y_val_mb = X_val_mb.to(device), y_val_mb.to(device)
                y_val_pred = model(X_val_mb)
                val_loss_mb = criterion(y_val_pred, y_val_mb)
                val_loss += val_loss_mb
            val_loss /= len(val_loader)
            learning_curve.append(val_loss)

        print(f'Epoch {epoch:03d}/{n_epochs} | Training loss: {training_loss:.6f} | Validation loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return learning_curve


resnet_optimizer = torch.optim.SGD(resnet34.parameters(), weight_decay=1e-4, momentum=0.9, lr=0.1)
resnet_scheduler = torch.optim.lr_scheduler.ExponentialLR(resnet_optimizer, gamma=0.1)
resnet_loss = nn.CrossEntropyLoss()
val_loss_curve = train_resnet(resnet34, cifar10_train_loader, cifar10_val_loader, resnet_optimizer, resnet_scheduler, resnet_loss, 500)

model_data = {
    'model_state_dict': resnet34.state_dict(),
    'optimizer_hyperparameters': {'weight_decay': 1e-4,
                                  'momentum': 0.9,
                                  'lr': 0.001},
    'scheduler_hyperparameters': {'gamma': 0.1,},
}
model_path = 'models/resnet34.pt'
torch.save(model_data, model_path)