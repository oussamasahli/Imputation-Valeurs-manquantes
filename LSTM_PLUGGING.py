# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# IMPORT DES LIBRAIRIES


import numpy as np
import math
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from functools import partial 


from torch.utils.data import DataLoader, RandomSampler
from hyperimpute.plugins.imputers import ImputerPlugin
import LSTM_auto_encoder
from utils import MyDataset
from hyperimpute.plugins.utils.simulate import simulate_nan
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")




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


                            # PLUGGING D'IMPUTATION


###############################################################################



eps = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Imputer REMASKER

class LSTM_Plugging(ImputerPlugin):

    def __init__(self):
        
        super().__init__()
                           
        self._model = Pluggin_Imputation()

    @staticmethod
    def name():
        return 'lstm'

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []

    def _fit(self, X, *args, **kwargs):
        return self

    def _transform(self, X: pd.DataFrame):
        X = torch.tensor(X.values, dtype=torch.float32).to(device)
        self._model.fit(X)
        return self._model.transform(X).detach().cpu().numpy()




# -----------------------------------------------------------------------------
# Imputer Plugin REMASKER 

class Pluggin_Imputation:

    def __init__(self):
        
        # ---------------------------------------------------------------------
        # Paramètre du pluggin d'Imputation 
        

        self.batch_size = 64
        self.norm_field_loss = False # Standardise les donénes prédites par REMASKER
        self.lr = 0.001             # Pas d'apprenitssage (learning rate)
        self.max_epochs = 600

        
        # ---------------------------------------------------------------------
        # Paramètre du modèle (auto_encoder)
        
        self.auto_encoder = None
        self.embed_dim = 32
        self.n_features = 1

        

    # -------------------------------------------------------------------------
    # Fit
    
    def fit(self, data):                      
       
        X = data.clone()
        X = X.cpu()
        seq_len = X.shape[1]
        
        
        # ---------------------------------------------------------------------
        # Normalisation Min-Max
        
        self.scaler = MinMaxScaler()                    
        X = self.scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)
                 
        
        # ---------------------------------------------------------------------
        # Mask : 0 si donnée manquante, 1 sinon
        
        M = 1 - (1 * (np.isnan(X)))                             
        M = M.float().to(device)                                                                      


        # ---------------------------------------------------------------------
        # Met à 0 les valeurs null de X (pour pouvoir appliquer l'auto encodage)
        
        X = torch.nan_to_num(X)                                                                                    
        X = X.to(device)
  
        
        # ---------------------------------------------------------------------
        # Auto-Encoder
        
        self.auto_encoder =  LSTM_auto_encoder.LSTM_AUTO_ENCODER(
            seq_len=seq_len, n_features=self.n_features, embedding_dim= self.embed_dim
        )
        
        self.auto_encoder.to(device)

        
        # ---------------------------------------------------------------------
        # Optimiseur
        
        optimizer = torch.optim.AdamW(self.auto_encoder.parameters(), lr=self.lr, betas=(0.9, 0.95))   
        
        
        
        # ---------------------------------------------------------------------
        # Dataset et DataLoader
        
        dataset = Dataset_LSTM(X)                                                                                                                                    
        dataloader = DataLoader( dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size)
        

        

        # -------------------------------------------------------------------------
        # Mode Train de l'auto_encoder
            
        self.auto_encoder.train()                                                    
        print("Lancement de l'apprentissage de LSTM_Plugging \n")
        
        
        # ---------------------------------------------------------------------
        # Apprentissage pour l'auto_encoder (apprend à déterminer une représentation de X)
        # Descente dde gradient par Batch
        
        for epoch in range(self.max_epochs):                                  

            optimizer.zero_grad()
            total_loss = 0
            
            # -----------------------------------------------------------------
            # Ittérations sur le batch d'exemple et les masques associés
            
            for i, x in enumerate(dataloader):              
                
                
                x = x.to(device, non_blocking=True)
                
            # ---------------------------------------------------------
            # Calcul de la RMSE
                    
                loss,pred = self.auto_encoder(x)
                    
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                
            
            #------------------------------------------------------------------
            # Loss moyenne (RMSE) à la fin de l'époch
            
            total_loss = math.sqrt ( total_loss / (i+1) ) 


            #------------------------------------------------------------------
            # Affiche de la Loss   
                                   
            if(epoch % 100 == 0):
                print("epoch :",epoch, ' - Loss :',total_loss )
            
          
            
        # -------------------------------------------------------------------------
        # Fin apprentissage REMASKER

        print("\n"+"Fin de l'apprentissage de LSTM \n")

        # On retourne l'auto encoder optimisé à la fonction transform (phase d'imputation)
        return self

   



    # -------------------------------------------------------------------------
    # Transform (Application du modèle optimisé)
    
    def transform(self, data):
        
        X = data.clone()
        no, dim = X.shape
        X = X.cpu()
        
        # -------------------------------------------------------------------------
        # Normalisation Min-Max
                         
        X = self.scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)


        # -------------------------------------------------------------------------
        # Mask : 0 si donnée manquante, 1 sinon
        
        M = 1 - (1 * (np.isnan(X)))  


        # -------------------------------------------------------------------------
        # Met à 0 les valeurs null de X (pour pouvoir appliquer l'auto encodage)     
                             
        X = np.nan_to_num(X)                                                   

        X = torch.from_numpy(X).to(device)
        M = M.to(device)
        
        
        # -------------------------------------------------------------------------
        # Mode Eval de l'auto_encoder
        
        self.auto_encoder.eval()                                                     
  

        # -------------------------------------------------------------------------
        # # Prédiction de l'auto_encoder pour chaque entrée de X

    
        with torch.no_grad():

            _, pred = self.auto_encoder(X)
            imputed_data = pred.squeeze(dim=2)

            
            
        # -------------------------------------------------------------------------
        # Normalisation Min-Max
                        
        imputed_data = self.scaler.fit_transform(imputed_data)
        imputed_data = torch.tensor(imputed_data, dtype=torch.float32)

        M = M.cpu()
        
        # -------------------------------------------------------------------------
        # Ajoute les prédictions seulement pour les données  manquantes
        imputed_data = imputed_data.detach().cpu()
        return M * np.nan_to_num(data.cpu()) + (1 - M) * imputed_data         

    
    # -------------------------------------------------------------------------
    # Fit_Transform / Entrée : Dataset X avec des données manquantes et retourne le dataset imputé
    def fit_transform(self, X):                  
        return self.fit(X).transform(X)                                        





################################################################################


                                # Main TEST
                                

#################################################################################


if __name__ == '__main__':

    from hyperimpute.plugins.imputers import Imputers


    # -------------------------------------------------------------------------
    # Data
    
    X, y = load_iris(as_frame=True, return_X_y=True)

    
    # -----------------------------------------------------------------
    # # Simulation données manquantes 
    # On injecte volontairement des donénse manquantes dans x
    
    X_ = simulate_nan( X.values, 0.1,  "MAR", sample_columns= True)
    X_miss = pd.DataFrame(X_["X_incomp"], columns=X.columns)
    
    
    # Test Remasker
    
    imputers = Imputers()
    imputers.add(LSTM_Plugging.name(), LSTM_Plugging)
    imputer = imputers.get('lstm')
    imputed_data = imputer.fit_transform(X_miss)











