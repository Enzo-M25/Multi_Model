 
import numpy as np

from Model_folder.Model import Model
from Model_folder.RL import RL
from Jauge import Jauge

class Choix :
    """
    Choix du / des meilleurs modele(s) selon differents criteres

    Attributs
    models : liste des modeles sur lesquels ont ete effectue une calibration
    """

    def __init__(self):
        self.models = [] 
        
    def add_model(self, model) -> None :
        """
        Ajoute model a la liste de modeles a considerer
        
        Paramètre d’entrée :
        model : une instance de Model
        """
        
        if isinstance(model, Model):
            self.models.append(model)
        else:
            raise TypeError("Seuls Model et ses sous-classes sont autorisés")
    
    
    def comparaison_models(self, fct_calib:str, dict_crit:dict[str,float]=None) -> list[Model] :
        """
        Compare les modeles presents dans la liste models et selectionne celui (ceux) dont le critere de calibration est le meilleur et respectant une cohérence avec la validation

        Paramètre d'entree :
        fct_calib : le critere qui a ete choisi pour calibrer les modeles
        dict_crit : (optionnel dans le cas d'un seul critere) dictionnaire des noms des criteres sur lesquels on effectue la calibration associes à leurs poids respectifs

        Paramètre de sortie :
        le modele de la liste ayant le meilleur critere
        """
        
        # Configuration pour chaque critère (à adapter selon vos besoins)
        criteria_config = {
            'crit_NSE': {'objective': 'maximize', 'threshold': 0.2},
            'crit_NSE_log': {'objective': 'maximize', 'threshold': 0.2},
            'crit_KGE': {'objective': 'maximize', 'threshold': 0.1},
            'crit_RMSE': {'objective': 'minimize', 'threshold': 0.5}, #TODO 
            'crit_Biais': {'objective': 'zero', 'threshold': 5},
        }

        # Vérification du critère

        if fct_calib in criteria_config :
            config = criteria_config[fct_calib]
        elif fct_calib == "crit_mix" and dict_crit is not None :
            name_crit = next(iter(dict_crit))
            config = criteria_config[name_crit]
        else :
            raise ValueError(f"Critère '{fct_calib}' non pris en charge ou pas de dicitonnaire fourni.")
        
        list_best_models = []
        best_value = -np.inf if config['objective'] == 'maximize' else np.inf
        best_index = -1

        for i, model in enumerate(self.models):
            # Vérification de la cohérence calibration/validation
            diff = abs(model.crit_calib - model.crit_valid)
            if diff > config['threshold']:
                continue

            # Évaluation de la performance
            current_value = model.crit_calib #TODO 0.5 * calib + 0.5 * valid ?
            if (config['objective'] == 'maximize' and current_value > best_value) or \
            (config['objective'] == 'minimize' and current_value < best_value) or \
            (config['objective'] == 'zero' and abs(current_value) < abs(best_value)):
                best_value = current_value
                best_index = i

        if best_index == -1:
            raise ValueError("Aucun modèle ne satisfait les critères de sélection.")
        
        for i, model in enumerate(self.models):
            # Vérification de la cohérence calibration/validation
            diff = abs(model.crit_calib - model.crit_valid)
            if diff > config['threshold']:
                continue

            # Évaluation de la performance
            current_value = model.crit_calib
            if (abs(best_value-current_value) < 0.05): #TODO valeur à adapter ?
                list_best_models.append(model)
        
        print(f"\n=== Modèle(s) sélectionné(s) : {[m.nom_model for m in list_best_models]} === \n")
        return list_best_models
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def comparaison_models_old(self, fct_calib:str, dict_crit:dict[str,float]=None) -> Model :
        """
        Ancienne version
        """
        
        # Configuration pour chaque critère (à adapter selon vos besoins)
        criteria_config = {
            'crit_NSE': {'objective': 'maximize', 'threshold': 0.1},
            'crit_NSE_log': {'objective': 'maximize', 'threshold': 0.1},
            'crit_KGE': {'objective': 'maximize', 'threshold': 0.1},
            'crit_RMSE': {'objective': 'minimize', 'threshold': 1}, #TODO 
            'crit_Biais': {'objective': 'zero', 'threshold': 5},
        }

        # Vérification du critère

        if fct_calib in criteria_config :
            config = criteria_config[fct_calib]
        elif fct_calib == "crit_mix" and dict_crit is not None :
            name_crit = next(iter(dict_crit))
            config = criteria_config[name_crit]
        else :
            raise ValueError(f"Critère '{fct_calib}' non pris en charge ou pas de dicitonnaire fourni.")
        
        best_value = -np.inf if config['objective'] == 'maximize' else np.inf
        best_index = -1

        for i, model in enumerate(self.models):
            # Vérification de la cohérence calibration/validation
            diff = abs(model.crit_calib - model.crit_valid)
            if diff > config['threshold']:
                continue

            # Évaluation de la performance
            current_value = model.crit_calib
            if (config['objective'] == 'maximize' and current_value > best_value) or \
            (config['objective'] == 'minimize' and current_value < best_value) or \
            (config['objective'] == 'zero' and abs(current_value) < abs(best_value)):
                best_value = current_value
                best_index = i

        if best_index == -1:
            raise ValueError("Aucun modèle ne satisfait les critères de sélection.")
        
        print(f"\n=== Modèle sélectionné : {self.models[best_index].nom_model} ===")
        return self.models[best_index]


