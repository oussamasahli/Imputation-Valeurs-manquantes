# -*- coding: utf-8 -*-


from hyperimpute.plugins.utils.simulate import simulate_nan
from sklearn.datasets import load_iris
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
import math
import copy
import torch.nn.functional as F
from functools import partial
from utils import Embedding, PositionalEncoding
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")



eps = 1e-6
                                           
     
      
    
###############################################################################


                                # ATTENTION MULTI TETE
                                
                    
###############################################################################

"""

L'objet d'attention multi tête se compose de plusieurs couche d'attention fonctionnant en parralèle

"""

class Attention(nn.Module):
    
    def __init__(self, num_heads, dim, dropout = 0.1):
        
        super().__init__()
        
        self.dim = dim
        self.d_k = dim // num_heads
        self.h = num_heads
        
        self.q_linear = nn.Linear(dim,dim)
        self.v_linear = nn.Linear(dim,dim)
        self.k_linear = nn.Linear(dim,dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim,dim)
    
    
    def forward(self, q, k, v):
        
        # q :  Batch_size * seq_len * d_model
        
        # on divise les embedings en N têtes : 
        # q : batch_size * num_heads * seq_len * (d_model / N), où (d_model / N) = d_k
        
        k = self.k_linear(k).view(k.shape[0], -1, self.h, self.d_k)
        q = self.q_linear(q).view(q.shape[0], -1, self.h, self.d_k)
        v = self.v_linear(v).view(v.shape[0], -1, self.h, self.d_k)
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
    
        # Mécanisme d'attention
        # décrit par Attention(Q,K,V) = softmax( QK^T/ racine(d_k) ) * V
        
        # Q : Matrice contenant la requête (embedding d'une valeur de la séquence)
        # K : Toutes les clés (embeddins de toutes les valeurs de la séquence)
        # V : les valeurs (embeddins de toutes les valeurs de la séquence)
        
        # Ainsi, les valeurs en V sont multipliées avec des poids d'attention a
        # tel que a = softmax( QK^T/ racine(d_k) )

        
        # Les poids a sont définis par la façon  dont chaque valeur de la séquence (Q)
        # est influencé par toutes les autres valeurs de la séquence (K)
        
        a = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_k)

        # Softmax est applqiué aux poids a pour avoir une distribution comprise 
        # entre 0 et 1
        
        a = F.softmax(a, dim=-1)
 
            
        if self.dropout is not None:
            a = self.dropout(a)
        
        # On applique les poids à toutes les valeurs de la séquence (V)
        attention = torch.matmul(a, v)
        
        concat = attention.transpose(1,2).contiguous().view(q.shape[0], -1, self.dim)
        output = self.out(concat)
    
        return output
    
 
    
 
    
 ###############################################################################


                                 # MLP
                                 
                     
 ###############################################################################
    
    
    
class MLP(nn.Module):
    
    def __init__(self, d_model, hiden=100, dropout = 0.1):
        
        super().__init__() 

        self.l1 = nn.Linear(d_model, hiden)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(hiden, d_model)
        
    def forward(self, x):
        
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)
        return x





###############################################################################


                                # STANDARDISATION
                                
                    
###############################################################################


class Standardisation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        scaled_data = (x - means) / stds
        
        return scaled_data
        
    
        
    


###############################################################################


                                # ENCODEUR
                                
                    
###############################################################################


# -----------------------------------------------------------------------------
# Couches de l'encodeur

class EncoderLayer(nn.Module):
    
    def __init__(self, embed_dim,  num_heads ,mlp_lattent, dropout = 0.1):
        
        super().__init__()
        
        self.scaled= Standardisation()
        self.attn = Attention(num_heads, embed_dim)
        self.mlp = MLP(embed_dim, mlp_lattent)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        scaled = self.scaled(x)  
        x = x + self.dropout(self.attn(scaled,scaled,scaled))  
        
        scaled = self.scaled(x)
        x = x + self.dropout(self.mlp(scaled))
        
        return x
    
 
    
 # -----------------------------------------------------------------------------
 # Encodeur avec remasquage des données x
    
class Encoder(nn.Module):
    
    def __init__(self, dim, embed_dim, depth, num_heads,mlp_lattent, jeton_masquage):
        super().__init__()
        

        self.depth = depth
        self.jeton_masquage = jeton_masquage
        
        self.embed = Embedding(dim, embed_dim)   
        self.pe = PositionalEncoding(embed_dim, dim, cls_token=False )  
        
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(embed_dim, num_heads,mlp_lattent)) \
                                     for i in range(depth)]) 
        
        self.norm = Standardisation()
        
        
    def forward(self, x, miss_idx, mask_ratio=0.5) :
        
        x = self.embed(x)
        x = self.pe(x)
               
        
        # ---------------------------------------------------------------------
        # Maskage aléatoire  (seulement dans la phase training (ajustement) )
                                                                                                                                             
        N, L, D = x.shape  
            
        # On applique le remasquage de x seulement dans la phase d'ajustemnt
        # selon l'auteur, cette tâche de re-masquage encourage le modèle à 
        # apprendre des représentations invariantes par rapport aux données manquantes
        
        if self.training: 
            
            # Nombre d'élément de x que l'on garde , le reste est remasqué
            garde = int(L * (1 - mask_ratio)) 
            
            # On masque seulement les valeurs présentes dans x , pas les valeurs manquantes
            # Donc si on choisit par exmple de masquer 3 valeurs alors 
            # 2 sont présentes dans x , alors une valeur manquentes sera considéré comme re-masqué
            # et nous on ne veut pas ça
            
            garde = min( garde , int(torch.min(torch.sum(miss_idx, dim=1))) )
        
        else :
            # On garde les données observées , mais on doit avoir
            # le même nombre sur chaque exemple
            garde = int(torch.min(torch.sum(miss_idx, dim=1)))
           
            
        # Pour chaque exemple, on récupère les indices des éléments présents
        present = []        
        for m in miss_idx:
            present.append(np.where(m==1)[0]) 
                
        # Parmis les éléments présents de x , on choisit d'en re-masque garde au hasard
        # On commence par mélangé les indices
            
        for l in present:
            np.random.shuffle(l)
            
        indice_garde = []
        # et on en garde que 'garde'
        for p in present : 
            indice_garde.append(p[:garde])
            
        # On construit le dataframe masqué
        x_masked = torch.zeros(N,garde,D)
            
        for i in range(0,len(indice_garde)):
            x_masked[i]=x[i][indice_garde[i]]
               
        # On construit le masque de qui a permis d'obtenir x_masked à partir de x
            
        remask = torch.zeros(miss_idx.shape[0], miss_idx.shape[1])
            
        # les éléments conservés sont mis à 1 dans remask
        for i in range(0,len(indice_garde)):
            remask[i][indice_garde[i]] = 1
                
        x = x_masked

        
        # Fin du masquage aléatoire
        # ---------------------------------------------------------------------



        # ---------------------------------------------------------------------
        # Application d'un pos_embedding sur le jeton de masquage
        # Ajout du jeton de masquage à x avant l'encodage
        
        jeton_masquage = self.jeton_masquage + self.pe.pe[:, :1, :]               
        jeton_masquages = jeton_masquage.expand(x.shape[0], -1, -1) 
                     
        x = torch.cat((jeton_masquages, x), dim=1)               
        
        
        for i in range(self.depth):
            x = self.layers[i](x)

        

        return self.norm(x) , remask
 


 


###############################################################################


                                # DECODEUR
                                
                    
###############################################################################

   

# -----------------------------------------------------------------------------
# Couches du décodeur


class DecoderLayer(nn.Module):
    def __init__(self, decoder_embed_dim, num_heads,mlp_lattent, dropout=0.1):
        super().__init__()
        
        self.scaled = Standardisation()
        self.dropout = nn.Dropout(dropout)    
        self.attn = Attention(num_heads, decoder_embed_dim)
        self.mlp = MLP(decoder_embed_dim)
        
        
    def forward(self, x):
    
        scaled = self.scaled(x)
        x = x + self.dropout(self.attn(scaled,scaled,scaled))
        
        scaled = self.scaled(x)
        x = x + self.dropout(self.mlp(scaled))
        
        return x
    


# -----------------------------------------------------------------------------
# Décodeur avec jetons de masquage en début de séquence pour les prédictions

    
class Decoder(nn.Module):
    
    def __init__(self, dim,embed_dim, decoder_embed_dim, decoder_depth, num_heads, mlp_lattent):
        
        super().__init__()
        
        self.dim = dim
        self.embed_dim = embed_dim
        self.decoder_depth = decoder_depth
        self.embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)    
        self.pe = PositionalEncoding(decoder_embed_dim, dim, cls_token=True ) 
        
        self.layers = nn.ModuleList([copy.deepcopy(DecoderLayer(decoder_embed_dim, num_heads,mlp_lattent))\
                                     for i in range(decoder_depth)])
        
        self.norm = Standardisation()
        
        
    def forward(self, x_masked, m):
        
        x = self.embed(x_masked)
        
        N, L, D = x_masked.shape[0], self.dim ,  self.embed_dim                   

        
        # On ajoute le padding sur x_masqued pour retrouver la taille initial de x
        pad = torch.zeros(N,L,D)
        
        # On enlève le jeton de masquage de x_masked
        x_ = x_masked[:, 1:, :]
        
        # On ajoute le padding 0 au bon endroit
        present = []        
        for ma in m:
            present.append(np.where(ma==1)[0])

        for i in range(0,len(present)):
                pad[i][present[i]] = x_[i]


        # on remet le jeton de masquage au début
        x = torch.cat([x_masked[:, :1, :], pad], dim=1) 

        
        
        x = self.pe(x)
        
        for i in range(self.decoder_depth):
            x = self.layers[i](x)
            
        return self.norm(x)






###############################################################################


                                # TRANSFORMEUR
                                
                    
###############################################################################




class Transformer(nn.Module):
    def __init__(self, dim=4, embed_dim=8, depth=4, num_heads=4,
        decoder_embed_dim=8, decoder_depth=2, decoder_num_heads=1,mlp_lattent=100,
        norm_layer=partial(nn.LayerNorm, eps=eps), norm_field_loss=False):
        
        super().__init__()
        
        self.norm_field_loss = norm_field_loss
        self.jeton_masquage = nn.Parameter(torch.zeros(1, 1, embed_dim)) 
        self.encoder = Encoder(dim, embed_dim, depth, num_heads,mlp_lattent,self.jeton_masquage)
        self.decoder = Decoder(dim, embed_dim, decoder_embed_dim, decoder_depth, num_heads, mlp_lattent)
        self.out = nn.Linear(decoder_embed_dim, 1)
        
        
    def forward(self, x, miss_idx, mask_ratio = 0.5):
    
        
        e_outputs, m = self.encoder(x, miss_idx, mask_ratio)
        d_output = self.decoder(e_outputs, m)
      
        
        pred = torch.tanh(self.out(d_output)) / 2 + 0.5
        
        # ---------------------------------------------------------------------
        # On enlève le jetons de masquage qu' on avait mis pour l'encodage
        
        pred = pred[:, 1:, :] 
        

        #----------------------------------------------------------------------
        # Calcul de la Loss RMSE
        
        target = x.squeeze(dim=1)
        
        # ---------------------------------------------------------------------
        # Standardisation des données
        
        if self.norm_field_loss:                                                                              
            scaler = StandardScaler()
            target = scaler.fit_transform(target)
        
        
        
        
        #----------------------------------------------------------------------
        # Loss RMSE

        loss = (pred.squeeze(dim=2) - target) ** 2   

        
        I_remask = miss_idx-m # ceux qui était présent mais qui on été ensuite masqués
        I_unmask = miss_idx * m
                         
        
        # loss calculée sur les données remasquées présentes auparavant
        if(I_remask.sum() != 0):
            loss_re = (loss * I_remask).sum() / I_remask.sum()  
        else :
            loss_re = (loss * I_remask).sum() 
            
            
         # loss calculée sur les données obsérvées après le remasquage  
        if(I_unmask.sum() != 0):
            loss_m = (loss * I_unmask).sum() / I_unmask.sum()  
        else :
            loss_m = (loss * I_unmask).sum() 

    
        loss = loss_re + loss_m


        return loss, pred, m
        
    
    
    
    
    
    
    
###############################################################################


                                # MAIN TEST
                                

###############################################################################




# ---------------------------------------------------------------------------
# Modèle


model =  Transformer (dim=4, embed_dim=8, depth=4, num_heads=4,
    decoder_embed_dim=8, decoder_depth=2, decoder_num_heads=1,mlp_lattent=100,
    norm_layer=partial(nn.LayerNorm, eps=eps), norm_field_loss=False )



# -------------------------------------------------------------------------
# Data iris.csv

X, y = load_iris(as_frame=True, return_X_y=True)


# -----------------------------------------------------------------
# # Simulation données manquantes 
# On injecte volontairement des données manquantes dans x

X_ = simulate_nan( X.values, 0.1,  "MAR", sample_columns= True)
X_miss = pd.DataFrame(X_["X_incomp"], columns=X.columns)
X = torch.tensor(X_miss.values, dtype=torch.float32)


# -------------------------------------------------------------------------
# Mask de X , 0 si null dans X , 1 sinon

M = 1 - (1 * (np.isnan(X)))   

# -------------------------------------------------------------------------
# Remplit les valeurs null de x par 0                          
X = torch.nan_to_num(X)    

# Rajoute une dimension  pour pouvoir appliquer l'embedding                                       
X = X.unsqueeze(dim=1)                     
                            
                                                                 
loss, pred, m = model.forward(X, M, 0.5) 







    
    