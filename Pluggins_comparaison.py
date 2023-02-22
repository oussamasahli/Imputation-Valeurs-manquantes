# -*- coding: utf-8 -*-


# -----------------------------------------------------------------------------
# Import des librairies


import copy
from typing import Any
import warnings
import numpy as np
import pandas as pd


from hyperimpute.plugins.imputers import Imputers
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.distributions import enable_reproducible_results
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from utils import distance_Wasserstein
from LSTM_PLUGGING import LSTM_Plugging



# -----------------------------------------------------------------------------
# Choisit un seed aléatoire (utile pour le choix de nombre alétoire)

enable_reproducible_results()

# -----------------------------------------------------------------------------
# Ignorer les warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Nécessaire pour appeller définir l'imputeur avec la méthode choisit plus bas

imputers = Imputers()
imputers.add(LSTM_Plugging.name(), LSTM_Plugging)



###############################################################################


                    # COMPARAISON DES PLUGGINS D'IMPUTATION


################################################################################



def pluggins_comparaisons( remasker: Any,
                          X: pd.DataFrame,
                          y: pd.DataFrame,
                          methods_imputation: list = ["mean", "missforest", "ice", "gain"],
                          probas_absence: list = [0.1, 0.3, 0.5, 0.7],
                          scenarios: list = ["MAR", "MCAR", "MNAR"] ) :

    
    scenarios_imputation = {}
    
    
    # -------------------------------------------------------------------------
    # Normalisation des données
    
    scaler = MinMaxScaler()
    cols = X.columns
    X = pd.DataFrame(scaler.fit_transform(X), columns=cols)


    #--------------------------------------------------------------------------
    # Création de tous les scénarios d'absences de données
    
    for s in scenarios:
        for p in probas_absence:
            if s not in scenarios_imputation:
                scenarios_imputation[s] = {}
            
            # -----------------------------------------------------------------
            # Simulation de l'imputation du dataset
            # Pour chaque scénarion, on définit  un triplet :
            # (dataframe X, dataframe X avec des données manquantes, Mask indiquant les indices des données cachées)
            
            columns = X.columns
            sampled_columns = columns[list(range( len(columns) ))] 


            # -----------------------------------------------------------------
            # Simulation de l'imputation sur le dataframe X
            
            X_ = simulate_nan( X[sampled_columns].values, p, s, sample_columns= True)


            _mask = pd.DataFrame(X_["mask"], columns=sampled_columns)
            _x_miss = pd.DataFrame(X_["X_incomp"], columns=sampled_columns)

            mask = pd.DataFrame(np.zeros(X.shape), columns=columns)
            
            
            #------------------------------------------------------------------
            # Mask avec 1 si la donnée a été mise à nulle dans X , 0 sinon
            
            mask[sampled_columns] = pd.DataFrame(_mask, columns=sampled_columns)  

            X_miss = pd.DataFrame(X.copy(), columns=columns) 
            X_miss[sampled_columns] = _x_miss

            scenarios_imputation[s][p] = ( pd.DataFrame(X, columns=columns), X_miss, mask )
            
            
    #--------------------------------------------------------------------------
    # Calcul de la RMSE , WD, AUROC à partir de chaque méthode d'imputation
    # ,et pour chaque scénarion d'absence

    res_rmse = {}
    res_WD = {}
    res_auroc = {}
    
   
    for s in scenarios:
        res_rmse[s] = {}
        res_WD[s] = {}
        res_auroc[s] = {}
        
        for p in probas_absence:
            res_rmse[s][p] = {}
            res_WD[s][p] = {}
            res_auroc[s][p] = {}
            
    
            try: 
                x, x_miss, mask = scenarios_imputation[s][p]  
                
            
                #--------------------------------------------------------------
                # CALCUL RMSE, WD, AUROC sur notre modèle REMASKER
                
                # -------------------------------------------------------------
                # Imputation du jeu de donnée X_miss avec REMASKER
                
                imputed = remasker.fit_transform(x_miss.copy()) 
     

                # -------------------------------------------------------------
                # Calcul AUROC 
                # l'Apprentissage se fait sur le datarame avec les données imputés
                
  
                if len(np.unique(y)) > 20:
                    auroc_score = 0
                else:
                    clf = LogisticRegression(solver="liblinear", random_state=0).fit(np.asarray(imputed), np.asarray(y)) 
                    if len(np.unique(np.asarray(y))) > 2:
                        auroc_score = roc_auc_score(np.asarray(y), clf.predict_proba(np.asarray(imputed)), multi_class='ovr')
                    else:
                        auroc_score = roc_auc_score(np.asarray(y), clf.predict_proba(np.asarray(imputed))[:,1])
                
 
                #--------------------------------------------------------------
                # Calcul de la distance de Wasserstein (WD)
                
                distribution_score = distance_Wasserstein(imputed, x)
                
                # Calcul RMSE
                rmse_score = RMSE(np.asarray(imputed), np.asarray(x), np.asarray(mask)) 

                # -------------------------------------------------------------
                # Enregistrement des résultats 
                res_rmse[s][p]["remasker"] = rmse_score
                res_WD[s][p]["remasker"] = distribution_score
                res_auroc[s][p]["remasker"] = auroc_score 
                
                
                #--------------------------------------------------------------
                # Calcul RMSE, WD, AUROC pour les autres pluggins d'imputation
                
                    
                for method in methods_imputation: 
                    
                    #----------------------------------------------------------
                    # On impute le jeu de données avec la méthode choisit
                    
                    imputer = imputers.get(method) 
                    imputed = imputer.fit_transform(x_miss.copy()) 
                    
                    # ---------------------------------------------------------
                    # Calcul AUROC 
                    # l'Apprentissage se fait sur le datarame avec les données imputés
                    
                    if len(np.unique(y)) > 20:
                        auroc_score = 0
                    else:
                        clf = LogisticRegression(solver="liblinear", random_state=0).fit(np.asarray(imputed), np.asarray(y)) 
                        if len(np.unique(np.asarray(y))) > 2:
                            auroc_score = roc_auc_score(np.asarray(y), clf.predict_proba(np.asarray(imputed)), multi_class='ovr')
                        else:
                            auroc_score = roc_auc_score(np.asarray(y), clf.predict_proba(np.asarray(imputed))[:,1])
    
    
                    #----------------------------------------------------------
                    # Calcul de la distance de Wasserstein (WD)
                    
                    distribution_score = distance_Wasserstein(imputed, x) 
                    
                    
                    # ---------------------------------------------------------
                    # Calcul RMSE
                    rmse_score = RMSE(np.asarray(imputed), np.asarray(x), np.asarray(mask)) 
                        
                    res_rmse[s][p][method] = rmse_score
                    res_WD[s][p][method] = distribution_score
                    res_auroc[s][p][method] = auroc_score
  
            except BaseException as erreur:
                print(erreur)
                continue
    
    
    
   #---------------------------------------------------------------------------
   # Résultat
   
    rmse_res = []
    wd_res = []
    auroc_res = []
    
    for s in scenarios:
        for p in probas_absence:
            
            # -----------------------------------------------------------------
            # RMSE
            a=[s,p]
            l=list(res_rmse[s][p].values())
            rmse_res.append(a+l)
            
            # -----------------------------------------------------------------
            #WD
            l=list(res_WD[s][p].values())
            wd_res.append(a+l)
            
            # -----------------------------------------------------------------
            # AUROC
            l=list(res_auroc[s][p].values())
            auroc_res.append(a+l)
   
    

    
    headers = ( ["Scenario", "proba_absence"]
            + [f"Imputer: {remasker.name()}"]
            + methods_imputation )

    sep = "\n==========================================================\n\n"
    
    print("RMSE score")
    rmse = pd.DataFrame(rmse_res, columns=headers)
    print(rmse)

    print(sep + "Wasserstein score")
    wd = pd.DataFrame(wd_res, columns=headers)
    print(wd)

    print(sep + "AUROC score")
    auroc = pd.DataFrame(auroc_res, columns=headers)
    print(auroc)


    return { "headers": headers, "rmse": rmse, "wasserstein": wd, 'auroc': auroc }
                
                
                
                
                
                
                
                