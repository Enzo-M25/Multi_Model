  
import pandas as pd
import numpy as np

from Critereprev import Critereprev
from Jauge import Jauge
from Model_folder import Model
from Model_folder.RL import RL
from Model_folder.GR4J import GR4J
from Model_folder.HydroModPy import HydroModPy
from Choix import Choix
from Pre_process import Pre_Process
from Post_process import Outputs

import os
from os.path import dirname, abspath
from datetime import datetime, date


def apply_transform(obs:np.ndarray, sim:np.ndarray, mode:str, model:Model) -> tuple[np.ndarray, np.ndarray] :
    """
    Retourne obs_t, sim_t selon mode (None, "log", "inv")

    Paramètres d'entrée :
    obs : Valeur des débits observées
    sim : Valeur des débits simulés
    mode : transformation à appliquer
    model : Le modèle sur lequel on effectue le calcul du critère

    Paramètre de sortie :
    obs-f, sim_f : Débits observés et simulés après application de la transformation
    """
    obs_f = obs.astype(float)
    sim_f = sim.astype(float)
    if mode == "log":
        Qbar = obs_f.mean()
        eps = 0 if model.nom_model == "GR4J" else Qbar / 100
        return np.log(obs_f + eps), np.log(sim_f + eps)
    elif mode == "inv":
        return 1 / obs_f, 1 / sim_f
    else:
        return obs_f, sim_f

def critere_prevision(model: Model, Q_obs: np.ndarray, Q_sim: np.ndarray, fct_calib: str, transfo: list[str], dict_crit: dict[str, float]) -> float:
    """
    Calcule la valeur du critère choisi pour un model et une transformation des débits donnés

    Paramètres d'entrée :
    model : Le modèle sur lequel on effectue le calcul du critère
    Q_obs : Valeur des débits observées
    Q_sim : Valeur des débits simulés
    fct_calib : Critère à calculer
    transfo : Transformation à appliquer sur les débits
    dict_crit : Dictionnaire donnant le nom et le poids des critères sélectionnés dans un cas où l'on fait le choix d'un critère multiple

    Paramètre de sortie :
    crit : Valeur du critère calculée
    """

    crit = 0.0

    if fct_calib == "crit_mix":
        total_weight = sum(dict_crit.values())
        if total_weight == 0:
            raise ValueError("La somme des poids est nulle, impossible de normaliser")

        for idx, (crit_name, weight) in enumerate(dict_crit.items()):
            mode = transfo[idx] or None
            obs_t, sim_t = apply_transform(Q_obs, Q_sim, mode, model)
            cp = Critereprev(obs_t, sim_t)
            value = getattr(cp, crit_name)()
            crit += weight * value

        return crit / total_weight

    else:
        mode = transfo[0] or None
        obs_t, sim_t = apply_transform(Q_obs, Q_sim, mode, model)
        cp = Critereprev(obs_t, sim_t)
        return getattr(cp, fct_calib)()

def parse_date(d:str) -> datetime :
    """
    Vérifie que le format de date rentré par l'utilisateur est correct (YYYY-MM-DD)
    
    Paramètre d'entrée :
    d : la str correspondant à la date
    
    Paramètre de sortie :
    La date au bon format ou une erreur sinon
    """
    
    try:
        return datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Format invalide pour la date : {d} (attendu YYYY-MM-DD)")

def main():
    
    nom = "Test" ##########
    id = "J836311001" #########
    x = 299970 #########
    y = 6779720 #########
    num_dep = 56

    # Initialisation des jeux de données

    dossier = "C:\\Users\\enzma\\Documents\\rennes 1\\M2\\Semestre 2\\Stage\\codes_matlab_resev_lin\\stations"
    fichier = f"CAMELS_FR_tsd_{id}.csv"

    watershed = Pre_Process(
        example_path=r"C:\Users\enzma\Documents\Tests_Modeles\Test_Multi_Modeles - Copie\Multi_model\HydroModPy_functions",
        data_path=r"C:\Users\enzma\Documents\HydroModPy\Enzo\data",
        results_path= r"C:\Users\enzma\Documents\HydroModPy\Enzo\results",
        basin_name=nom,
        departement=num_dep,
        x=x,
        y=y, 
        dem_raster=r"C:\Users\enzma\Documents\HydroModPy\Enzo\data\regional dem.tif",
        hydrometry_csv= f"{id}_QmnJ(n=1_non-glissant).csv",
        year_start=2000,
        year_end=2020,
        example_year=2010
    )

    bv = Jauge(id, nom, dossier, fichier, watershed)

    # watershed.pre_processing()

    # Paramètres de la calibration

    fct_calib = "crit_NSE"

    transfo = ["log"]
    dict_crit = {"crit_KGE": 0.5, "crit_NSE": 0.5}

    t_calib_start = parse_date("2005-01-01")  ############
    t_calib_end = parse_date("2010-12-31")
    t_valid_start = parse_date("2010-01-01")
    t_valid_end = parse_date("2020-12-31")
    t_prev_start = parse_date("2021-01-01") # pour hydromodpy l'année doit être complète pour les comparaisons
    t_prev_end = parse_date("2021-12-31")

    if t_calib_start > t_calib_end or t_valid_start > t_valid_end or t_prev_start > t_prev_end :
        raise ValueError(f"Format invalide pour une période, début et fin inversé")

    mac = Choix()


    # Réservoir linéaire
    
    model1 = RL(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
    model1.param_calib(bv)
    print("\n=== Résultats du modèle de Résevoir linéaire (RL) ===")
    print(f"\n résultats calculés avec le(s) critère(s) : {fct_calib} et une transformation : {transfo}")
    print(f"  Alpha      : {model1.alpha}")
    print(f"  Vmax       : {model1.Vmax}")
    print(f"  {fct_calib} Calib  : {model1.crit_calib:.4f}")
    print(f"  {fct_calib} Valid  : {model1.crit_valid:.4f}")
    print("===============================\n")
    mac.add_model(model1)

    ### GR4J

    model2 = GR4J(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
    model2.param_calib(bv)
    print("\n=== Résultats du modèle GR4J ===")
    print(f"\n résultats calculés avec le(s) critère(s) : {fct_calib} et une transformation : {transfo}")
    print(f"{fct_calib} calibration : {model2.crit_calib:.4f}")
    print(f"{fct_calib} validation : {model2.crit_valid:.4f}")
    print("Paramètres calibrés :")
    for i, val in enumerate(model2.x, start=1):
        print(f"  X{i} : {val}")
    print("===============================\n")
    mac.add_model(model2)
    
    ### HYDROMODPY

    # model3 = HydroModPy(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib, r"C:\Users\enzma\Documents\Tests_Modeles\Test_Multi_Modeles - Copie\Multi_model\HydroModPy_functions",
    #                     'M', r"C:\Users\enzma\Documents\HydroModPy\Enzo\data\Meteo\REA", dict_crit=None)
    # model3.param_calib(bv)
    # print("\n=== Résultats du modèle HydroModPy ===")
    # print(f"\n résultats calculés avec le(s) critère(s) : {fct_calib} et une transformation : {transfo}")
    # print(f"{fct_calib} calibration : {model3.crit_calib:.4f}")
    # print(f"{fct_calib} validation : {model3.crit_valid:.4f}")
    # print("Paramètres calibrés :")
    # print(f"  Sy      : {model3.sy}")
    # print(f"  hk(m/s) : {model3.hk}")
    # print("===============================\n")
    # mac.add_model(model3)

    try :
        best = mac.comparaison_models(fct_calib) # best est une liste de model

        for i, model in enumerate(best):

            main_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(main_dir, f"figures_{nom}_{t_prev_start.year}_{t_prev_end.year}")
            os.makedirs(figures_dir, exist_ok=True)

            print(f"Prévision num {i+1} avec le modèle {model.nom_model} :\n")

            d, Q_sim = model.prevision(bv)

            # Affichage seul de la prévision

            result = Outputs(id,nom,figures_dir,d,Q_sim)
            result.affiche()

            if (len(Q_sim) == len(bv.serie_debit(t_prev_start,t_prev_end))) :
                Q_obs = bv.serie_debit(t_prev_start,t_prev_end)
            elif (len(Q_sim) == len(bv.serie_debit_mensuel(t_prev_start,t_prev_end))) :
                Q_obs = bv.serie_debit_mensuel(t_prev_start,t_prev_end)
            else :
                raise ValueError("Impossible d'afficher une comparaison simulé / observé. Pas assez de mesures de débits observées.")

            # Affichage de la prévision et des débits observés

            result_compar = Outputs(id,nom,figures_dir,d,Q_sim,Q_obs)
            result_compar.affiche()
            result_compar.affiche_nuage()

            # Critère de prévision

            print(f'{fct_calib} : {critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)}')

    except ValueError as e :
        print(f"Erreur lors de la sélection du modèle : {e}")
    
if __name__ == "__main__":
    main()
