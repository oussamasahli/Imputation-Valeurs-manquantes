# -*- coding: utf-8 -*-


# -----------------------------------------------------------------------------
# Import des librairies 


from Pluggins_comparaison import pluggins_comparaisons
from hyperimpute.plugins.imputers import Imputers
from REMASKER import REMASKER
import time
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, fetch_california_housing
import warnings
warnings.filterwarnings("ignore")




# -----------------------------------------------------------------------------
# Datasets disponibles

datasets = [ 'iris',  'obesity','climate', 'bike', 'compression', 'wine', 'yacht', 'spam', 'letter',
            'credit', 'raisin', 'california', 'diabetes']


# -----------------------------------------------------------------------------
# Différents pluggins d'imputation de données 

methods = ['lstm','gain', 'ice', 'mice', 'missforest', 'sinkhorn', 'miwae', 'miracle',
           'EM', 'mean', 'median', 'most_frequent', 'softimpute']




# -----------------------------------------------------------------------------
# Comparaison des différents pluggins d'imputation

Metriques_compared = {}
path = "C:/Users/osahl/Documents/AMAL_2023/EXPOSER/Code/data/"

for name in datasets[:4] :
    
    # -------------------------------------------------------------------------
    # Data
    
    if name in ['climate', 'compression', 'wine', 'yacht', 'spam', 'letter', 
                   'credit', 'raisin', 'bike', 'obesity', 'airfoil', 'blood',
                   'yeast', 'health', 'review', 'travel']:
    
        df = pd.read_csv(path + name + '.csv')
        last_col = df.columns[-1]
        
        y = df[last_col]
        X = df.drop(columns=[last_col])
    
    # Car dans data , la colonne y de ces dataframe ne contient pas des eniters mais des chiffres flottant
    # ce qui pose problème pour le calcul de l'auroc
    
    elif name == 'california':
        X, y = fetch_california_housing(as_frame=True, return_X_y=True)
        
    elif name == 'diabetes':
        X, y = load_diabetes(as_frame=True, return_X_y=True)
        
    elif name == 'iris':
        X, y = load_iris(as_frame=True, return_X_y=True)

    # -------------------------------------------------------------------------
    # Initialisation de l'imputer REMASKER avec les arguments prévus pour le dataset

    imputers = Imputers()
    imputers.add(REMASKER.name(), REMASKER)
    remasker = imputers.get('remasker')



    # Calcul des métriques 
    
    print('Dataset :', name)
    start = time.time()
    
    Metriques_compared[name] = pluggins_comparaisons( remasker = remasker,
                                  X=X, y=y,
                                  methods_imputation = methods,
                                  scenarios = ["MAR", "MCAR", "MNAR"],
                                  probas_absence = [0.1, 0.3, 0.5, 0.7] )


    end = time.time()
    elapsed = end - start
    print('temps exécution : ', (end - start)/60, ' min', "\n")

    #--------------------------------------------------------------------------
    # Ecriture des résultats
    
    Metriques_compared[name]['rmse'].to_csv("rmse_"+name+".csv")
    Metriques_compared[name]['wasserstein'].to_csv("wasserstein_"+name+".csv")
    Metriques_compared[name]['auroc'].to_csv("auroc_"+name+".csv")
    
    
    
    
    
    




