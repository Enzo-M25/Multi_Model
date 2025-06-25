
import subprocess
import os
import sys
from typing import Optional

from .Model import Model

class HydroModPy(Model):
    """
    Classe fille de Model
    Modele HydroModPy 

    Attributs
    t_calib : période de calibration du modèle
    t_valid : période de validation des débits
    t_prev : période de prévision des débits
    transfo : liste contenant les transformations appliquees aux debits (ie. "", "log", "inv")
    fct_calib : nom du critère sur lequel on effectue la calibration (NSE, NSE-log, KGE, RMSE, Biais)
    dict_crit : (optionnel dans le cas d'un seul critere) dictionnaire des noms des criteres sur lesquels on effectue la calibration associes à leurs poids respectifs
    crit_calib : meilleure valeur du critere de calibration obtenue lors de la calibration de celui-ci
    crit_valid : valeur du critere de validation obtenue lors de la validation de celui-ci
    nom_model : nom du modele (GR4J)
    sy : paramètre du modèle, coefficient de porosité
    hk : paramètre du modèle, conductivité hydraulique
    """

    def __init__(self, t_calib_start:str, t_calib_end:str, t_valid_start:str, t_valid_end:str, t_prev_start:str, t_prev_end:str,
                 transfo:list[str], fct_calib:str, dict_crit: Optional[dict[str, float]] = None) :
        
        super().__init__(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib, dict_crit)
        self.nom_model = "HydroModpy"
        self.sy: float | None = None
        self.hk: float | None = None
