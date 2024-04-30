## IMPORTING REQUIRED LIBRARIES
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim

## DATA LOADER CLASS

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

## EARLY STOPPING CLASS
# Early stopping class implementation in PyTorch
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


## MODEL CLASS

class AudioClassifier_simple(nn.Module):
    def __init__(self):
        super(AudioClassifier_simple, self).__init__()
        # Assuming the number of input features per sample is 2376 (from your error message input shape [32, 1, 2376])
        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, padding='same')  # Change input channels to 1
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool1d(kernel_size=8)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding='same')
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        # Calculate the number of features coming out of the last Conv/Pool layer to correctly set the input features for the dense layer
        self.fc1 = nn.Linear(self._get_conv_output((1, 2376)), 7)  # Output size needs to be calculated
        self.act3 = nn.Softmax(dim=1)

    def _get_conv_output(self, shape):
        input = torch.rand(*shape)
        output = self.pool1(self.drop1(self.act1(self.conv1(input))))
        output = self.drop2(self.act2(self.conv2(output)))
        return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act3(x)
        return x
