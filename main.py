  
import pandas as pd
import numpy as np

from config import (
    HYDROMODPY_FUNCTIONS,
    DATA_DIR,
    METEO_DIR,
    RESULTS_DIR,
    DEM_FILE,
    STATIONS_DIR,
    PARAM_CONFIG,
    WATERSHED_CONFIG
)

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

def generate_plots(model: Model, bv: Jauge, nom: str, id: str, t_prev_start: datetime, 
                  t_prev_end: datetime) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère les graphiques de prévision pour un modèle donné
    
    Paramètres d'entrée:
    model: Le modèle utilisé pour la prévision
    bv: L'objet Jauge représentant le bassin versant
    nom: Nom du bassin versant
    id: Identifiant du bassin versant
    t_prev_start: Date de début de prévision
    t_prev_end: Date de fin de prévision
    
    Paramètres de sortie :
    Q_obs, Q_sim: Les débits observés et simulés
    """
    # Création du répertoire pour les figures
    main_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(main_dir, f"figures_{nom}_{t_prev_start.year}_{t_prev_end.year}")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Prévision avec le modèle {model.nom_model} :\n")

    # Calcul de la prévision
    d, Q_sim = model.prevision(bv)

    # Vérification de la compatibilité des tailles
    if not (len(d) == len(Q_sim)):
        raise ValueError(
            f"Incompatibilité des tailles : dates ({len(d)}), Q_sim ({len(Q_sim)})"
        )

    # Affichage seul de la prévision
    result = Outputs(id, nom, figures_dir, d, Q_sim)
    result.affiche()

    # Récupération des débits observés
    if len(Q_sim) == len(bv.serie_debit(t_prev_start, t_prev_end)):
        Q_obs = bv.serie_debit(t_prev_start, t_prev_end)
    elif len(Q_sim) == len(bv.serie_debit_mensuel(t_prev_start, t_prev_end)):
        Q_obs = bv.serie_debit_mensuel(t_prev_start, t_prev_end)
    else:
        raise ValueError("Impossible d'afficher une comparaison simulé / observé. Pas assez de mesures de débits observées.")

    # Vérification de la compatibilité des tailles
    if not (len(d) == len(Q_sim) == len(Q_obs)):
        raise ValueError(
            f"Incompatibilité des tailles : dates ({len(d)}), Q_sim ({len(Q_sim)}), Q_obs ({len(Q_obs)})"
        )
    
    # Affichage comparatif
    result_compar = Outputs(id, nom, figures_dir, d, Q_sim, Q_obs)
    result_compar.affiche()
    result_compar.affiche_nuage()

    return Q_obs, Q_sim

def main():
    
    nom = WATERSHED_CONFIG["nom"]
    id = WATERSHED_CONFIG["id"]
    x = WATERSHED_CONFIG["x"]
    y = WATERSHED_CONFIG["y"]
    num_dep = WATERSHED_CONFIG["num_dep"]

    # Initialisation des jeux de données

    fichier = f"CAMELS_FR_tsd_{id}.csv"

    watershed = Pre_Process(
        example_path=HYDROMODPY_FUNCTIONS,
        data_path=DATA_DIR,
        results_path=RESULTS_DIR,
        basin_name=nom,
        departement=num_dep,
        x=x,
        y=y, 
        dem_raster=DEM_FILE,
        hydrometry_csv=f"{id}_QmnJ(n=1_non-glissant).csv",
        year_start=2000,
        year_end=2020,
        example_year=2010
    )

    bv = Jauge(id, nom, STATIONS_DIR, fichier, watershed)

    watershed.pre_processing()

    # Paramètres de la calibration

    fct_calib = PARAM_CONFIG["fct_calib"]

    transfo = PARAM_CONFIG["transfo"]
    dict_crit = PARAM_CONFIG["dict_crit"] #{"crit_KGE": 0.5, "crit_NSE": 0.5}

    t_calib_start = parse_date(PARAM_CONFIG["t_calib_start"]) 
    t_calib_end = parse_date(PARAM_CONFIG["t_calib_end"])
    t_valid_start = parse_date(PARAM_CONFIG["t_valid_start"])
    t_valid_end = parse_date(PARAM_CONFIG["t_valid_end"])
    t_prev_start = parse_date(PARAM_CONFIG["t_prev_start"]) # pour hydromodpy l'année doit être complète pour les comparaisons
    t_prev_end = parse_date(PARAM_CONFIG["t_prev_end"])

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

    # model3 = HydroModPy(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib,
    #                       HYDROMODPY_FUNCTIONS, 'M', METEO_DIR, dict_crit=None)
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

    ### HYDROMODPY avec calibration sur le réseau hydro

    # model4 = HydroModPy(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib,
    #                     HYDROMODPY_FUNCTIONS, 'M', METEO_DIR, dict_crit=None)
    # model4.param_calib_reseau(bv)
    # print("\n=== Résultats du modèle HydroModPy avec calibration du réseau hydro ===")
    # print(f"\n résultats calculés avec le(s) critère(s) : {fct_calib} et une transformation : {transfo}")
    # print(f"{fct_calib} calibration : {model4.crit_calib:.4f}")
    # print(f"{fct_calib} validation : {model4.crit_valid:.4f}")
    # print("Paramètres calibrés :")
    # print(f"  Sy      : {model4.sy}")
    # print(f"  hk(m/s) : {model4.hk}")
    # print("===============================\n")
    # mac.add_model(model4)

    try:
        best = mac.comparaison_models(fct_calib)
        
        for i, model in enumerate(best):
            # Génération des graphiques et récupération des débits
            Q_obs, Q_sim = generate_plots(model, bv, nom, id, t_prev_start, t_prev_end)
            
            # Calcul et affichage du critère de prévision
            crit = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)
            print(f'{fct_calib} : {crit}')

    except ValueError as e:
        print(f"Erreur lors de la sélection du modèle : {e}")
    
if __name__ == "__main__":
    main()

# TODO rajouter un code pour permettre de refaire une prédiction sur une autre période sans recalculer les paramètres du modèle