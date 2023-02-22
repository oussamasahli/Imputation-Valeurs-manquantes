# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from hyperimpute.plugins.utils.simulate import simulate_nan
from sklearn.datasets import load_iris
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from utils import MyDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset
import math


import warnings
warnings.filterwarnings("ignore")





###############################################################################


                            # ENCODER
                            
###############################################################################





class Encoder(nn.Module):
    
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    
    self.lstm1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.lstm2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
    
  def forward(self, x):
      
    size = len(x)
    x = x.reshape((size, self.seq_len, self.n_features))
    x, (_, _) = self.lstm1(x)
    x, (hidden_n, _) = self.lstm2(x)
    return hidden_n.reshape((size,self.n_features, self.embedding_dim))







###############################################################################



                               # DECODER
                               
                               
##############################################################################


class Decoder(nn.Module):
    
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    
    self.lstm1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.lstm2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
    
  def forward(self, x):
      
    size = len(x)
    x = x.repeat(1, self.seq_len, self.n_features)
    x = x.reshape((size, self.seq_len, self.input_dim))
    
    x, (hidden_n, cell_n) = self.lstm1(x)
    x, (hidden_n, cell_n) = self.lstm2(x)
    x = x.reshape((size,self.seq_len, self.hidden_dim))
    return self.output_layer(x)




###############################################################################



                               # LSTM AUTO-ENCODER
                               
                               
##############################################################################

class LSTM_AUTO_ENCODER(nn.Module):
    
    def __init__(self, seq_len=4, n_features=1, embedding_dim=64):
        super(LSTM_AUTO_ENCODER, self).__init__()
        
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        
        hidden = self.encoder(x)
        pred = self.decoder(hidden) 
 
        scaler = StandardScaler()
        target = scaler.fit_transform(x)
        
        loss = (pred.squeeze(dim=2) - torch.tensor(target)) ** 2 
        loss = loss.sum()/x.shape[0]
        
        return loss,pred



###############################################################################


                            # Dataset 


###############################################################################


class Dataset_LSTM(Dataset):

    def __init__(self, X):        
         self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx]




###############################################################################



                               # MAIN
                               
                               
##############################################################################




X, y = load_iris(as_frame=True, return_X_y=True)

X_ = simulate_nan( X.values, 0.1,  "MAR", sample_columns= True)
X_miss = pd.DataFrame(X_["X_incomp"], columns=X.columns)
X = torch.tensor(X_miss.values, dtype=torch.float32)

M = 1 - (1 * (np.isnan(X)))                           
X = torch.nan_to_num(X)    



seq_len=4
n_features=1
embedding_dim=8


model = LSTM_AUTO_ENCODER(seq_len, n_features, embedding_dim)
loss, pred = model(X)




