# -*- coding: utf-8 -*-


# -----------------------------------------------------------------------------
# Import des librairies


import numpy as np
from torch.utils.data import Dataset
from scipy.stats import wasserstein_distance
import torch.nn as nn
import torch



###############################################################################


                                # EMBEDDING


###############################################################################


#------------------------------------------------------------------------------
# Exemple : 
    
# Entrée : X :(1, 1, dim) , Sortie : (1, dim, embed_dim)



class Embedding(nn.Module):
    def __init__(self, dim=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.dim = dim
        self.emb = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        
        x = self.emb(x)                                                        
        x = x.transpose(1, 2)                                                                                                                                                                                                        
        x = self.norm(x)
        return x



###############################################################################


                     # EMBEDDING POSITIONNEL  (basé sur la fréquence)
                     

###############################################################################


#------------------------------------------------------------------------------
# PE (pos,2i) = sin( pos /10000**(2i/d_model) )
# PE (pos,2i+1) = cos( pos /10000**(2i/d_model) )

"""
 
- "pos" est la position ou la valeur d'index du mot particulier dans la phrase. 
 
- "d" est la longueur/dimension maximale du vecteur qui représente un mot particulier 
    dans la phrase.
 
- "i" représente les indices de chacune des dimensions d'intégration positionnelle. Il indique également la fréquence. 
    Lorsque i=0, elle est considérée comme la fréquence la plus élevée et pour les valeurs suivantes, 
    la fréquence est considérée comme une amplitude décroissante.﻿"

"""


class PositionalEncoding(nn.Module):

    def __init__(self, d_model,  max_len=5000, cls_token=False):
        super(PositionalEncoding, self).__init__()

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        #---------------------------------------------------------------
        # 1/10000**(2i/d_model)
        
        div_term = 1/10000**(torch.arange(0, d_model, 2).float()/d_model)       
        
        #---------------------------------------------------------------
        # PE (pos,2i) = sin( pos /10000**(2i/d_model) )
        
        self.pe[:, 0::2] = torch.sin(position * div_term)     
        
        #---------------------------------------------------------------
        # PE (pos,2i+1) = cos( pos /10000**(2i/d_model) )
        
        self.pe[:, 1::2] = torch.cos(position * div_term)     
        
        if cls_token:
            self.pe = np.concatenate([np.zeros([1, d_model]), self.pe], axis=0)
        
        self.pe = torch.tensor(self.pe).unsqueeze(0).float()
        
    def forward(self, x):

        x = x + self.pe
        return x
    



###############################################################################


                            # Dataset 


###############################################################################


class MyDataset(Dataset):

    def __init__(self, X, M):        
         self.X = X
         self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]



###############################################################################


                    # METRIQUES


###############################################################################
 

# -----------------------------------------------------------------------------
# La distance de Wasserstein (WD) pour mesurer la correspondance entre la 
# distribution imputée et la distribution de la vérité du terrain.


def distance_Wasserstein(imputed, data) :
    
    wd = 0
    dim= data.shape[1]
    
    for i in range(0,dim):
        wd += wasserstein_distance(np.asarray(data)[:,i], np.asarray(imputed)[:,i])
    return wd


