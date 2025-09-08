  
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count, current_process

from config import (
    HYDROMODPY_FUNCTIONS,
    DATA_DIR,
    METEO_DIR,
    RESULTS_DIR,
    DEM_FILE,
    STATIONS_DIR
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

def critere_prevision(model: Model, Q_obs: pd.Series, Q_sim: pd.Series, fct_calib: str, transfo: list[str], dict_crit: dict[str, float]) -> float:
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

def make_empty_results(nom=None, station_id=None):
    """
    Retourne deux dictionnaires (res1, res2) contenant toutes les clés attendues,
    initialisées à None sauf 'nom' et 'station_id'.
    """
    res1 = {
        "nom"                 : nom,
        "station_id"          : station_id,
        "RL_best"             : None,
        "alpha_RL"            : None,
        "Vmax_RL"             : None,
        "calib_RL"            : None,
        "valid_RL"            : None,
        "prev_RL"             : None,
        "GR4J_best"           : None,
        "params_GR4J"         : None,
        "calib_GR4J"          : None,
        "valid_GR4J"          : None,
        "prev_GR4J"           : None,
        "HMP_best"            : None,
        "params_HMP"          : {"sy": None, "hk": None},
        "calib_HMP"           : None,
        "valid_HMP"           : None,
        "prev_HMP"            : None,
        "HMP_reseau_best"     : None,
        "params_HMP_reseau"   : {"sy": None, "hk": None},
        "calib_HMP_reseau"    : None,
        "valid_HMP_reseau"    : None,
        "prev_HMP_reseau"     : None,
    }

    res2 = {
        "nom"                 : nom,
        "station_id"          : station_id,
        "RL_best"             : None,
        "alpha_RL"            : None,
        "Vmax_RL"             : None,
        "calib_RL"            : None,
        "valid_RL"            : None,
        "prev_RL"             : None,
        "GR4J_best"           : None,
        "params_GR4J"         : None,
        "calib_GR4J"          : None,
        "valid_GR4J"          : None,
        "prev_GR4J"           : None,
        "HMP_reseau_best"     : None,
        "params_HMP_reseau"   : {"sy": None, "hk": None},
        "calib_HMP_reseau"    : None,
        "valid_HMP_reseau"    : None,
        "prev_HMP_reseau"     : None,
    }

    return res1, res2

dossier = STATIONS_DIR

fct_calib = "crit_NSE"

transfo = ["log"]
dict_crit = None

t_calib_start = parse_date("2005-01-01")
t_calib_end = parse_date("2010-12-31")
t_valid_start = parse_date("2010-01-01")
t_valid_end = parse_date("2020-12-31")
t_prev_start = parse_date("2021-01-01") # pour hydromodpy l'année doit être complète pour les comparaisons
t_prev_end = parse_date("2021-12-31")

def process_file(args):

    start = time.time()

    nom, id, x, y, departement = args
    
    res1, res2 = make_empty_results(nom, id)

    # Vérifier que le fichier existe
    fichier = f"CAMELS_FR_tsd_{id}.csv"
    fichier_path = os.path.join(STATIONS_DIR, fichier)
    if not os.path.exists(fichier_path):
        print(f"❌ Fichier manquant pour {nom}: {fichier_path}")
        return None

    proc = current_process().name
    pid  = os.getpid()
    print(f"[{proc} | PID {pid}] traite le bv {nom}")

    # Valeurs par défaut si jamais un critère n'est pas défini
    crit_prev_RL = crit_prev_GR4J = crit_prev_HMP = crit_prev_HMP_reseau = None
    best_RL = best_GR4J = best_HMP = best_HMP_reseau = "No"

    crit_prev_RL2 = crit_prev_GR4J2 = crit_prev_HMP_reseau2 = None
    best_RL2 = best_GR4J2 = best_HMP_reseau2 = "No"

    try :
    
        fichier = f"CAMELS_FR_tsd_{id}.csv"

        watershed = Pre_Process(
            example_path=HYDROMODPY_FUNCTIONS,
            data_path=DATA_DIR,
            results_path=RESULTS_DIR,
            basin_name=nom,
            departement=departement,
            x=x,
            y=y,
            dem_raster=DEM_FILE,
            hydrometry_csv=f"{id}_QmnJ(n=1_non-glissant).csv",
            year_start=2000,
            year_end=2021,
            example_year=2010
        )

        bv = Jauge(id, nom, dossier, fichier, watershed)

        # mac = Choix()
        mac2 = Choix()
        
        model1 = RL(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
        model1.param_calib(bv)
        # mac.add_model(model1)
        mac2.add_model(model1)
        
        model2 = GR4J(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
        model2.param_calib(bv)
        # mac.add_model(model2)
        mac2.add_model(model2)
        
        # model3 = HydroModPy(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib,
        #                     HYDROMODPY_FUNCTIONS, 'M', METEO_DIR, dict_crit=None)
        # model3.param_calib(bv)
        # mac.add_model(model3)

        model4 = HydroModPy(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib,
                            HYDROMODPY_FUNCTIONS, 'M', METEO_DIR, dict_crit=None)
        model4.param_calib_reseau(bv)
        # mac.add_model(model4)
        mac2.add_model(model4)

        # best = mac.comparaison_models(fct_calib) # best est une liste de model
        best2 = mac2.comparaison_models(fct_calib) # best2 est une liste de model 

        # for model in best:

        #     d, Q_sim = model.prevision(bv)

        #     if (len(Q_sim) == len(bv.serie_debit(t_prev_start,t_prev_end))) :
        #         Q_obs = bv.serie_debit(t_prev_start,t_prev_end)
        #     elif (len(Q_sim) == len(bv.serie_debit_mensuel(t_prev_start,t_prev_end))) :
        #         Q_obs = bv.serie_debit_mensuel(t_prev_start,t_prev_end)
        #     else :
        #         raise ValueError("Impossible d'afficher une comparaison simulé / observé. Pas assez de mesures de débits observées.")

        #     if model.nom_model == "RL" :
        #         best_RL = "Yes"
        #         crit_prev_RL = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)
        #     elif model.nom_model == "GR4J" :
        #         best_GR4J = "Yes"
        #         crit_prev_GR4J = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)
        #     elif model.nom_model == "HydroModpy" :
        #         best_HMP = "Yes"
        #         crit_prev_HMP = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)
        #     elif model.nom_model == "HydroModpy_reseau" :
        #         best_HMP_reseau = "Yes"
        #         crit_prev_HMP_reseau = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)

        for model in best2:

            d, Q_sim = model.prevision(bv)

            if (len(Q_sim) == len(bv.serie_debit(t_prev_start,t_prev_end))) :
                Q_obs = bv.serie_debit(t_prev_start,t_prev_end)
            elif (len(Q_sim) == len(bv.serie_debit_mensuel(t_prev_start,t_prev_end))) :
                Q_obs = bv.serie_debit_mensuel(t_prev_start,t_prev_end)
            else :
                raise ValueError("Impossible d'afficher une comparaison simulé / observé. Pas assez de mesures de débits observées.")

            if model.nom_model == "RL" :
                best_RL2 = "Yes"
                crit_prev_RL2 = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)
            elif model.nom_model == "GR4J" :
                best_GR4J2 = "Yes"
                crit_prev_GR4J2 = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)
            elif model.nom_model == "HydroModpy_reseau" :
                best_HMP_reseau2 = "Yes"
                crit_prev_HMP_reseau2 = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)

        cpu_time_used = time.time() - start

        # res1 = {
        #     "nom"         : nom,
        #     "station_id"  : id,
        #     "RL_best"     : best_RL,
        #     "alpha_RL"    : getattr(model1, "alpha", None),
        #     "Vmax_RL"     : getattr(model1, "Vmax", None),
        #     "calib_RL"    : getattr(model1, "crit_calib", None),
        #     "valid_RL"    : getattr(model1, "crit_valid", None),
        #     "prev_RL"     : crit_prev_RL,
        #     "GR4J_best"   : best_GR4J,
        #     "params_GR4J" : getattr(model2, "x", None),
        #     "calib_GR4J"  : getattr(model2, "crit_calib", None),
        #     "valid_GR4J"  : getattr(model2, "crit_valid", None),
        #     "prev_GR4J"   : crit_prev_GR4J,
        #     "HMP_best"    : best_HMP,
        #     "params_HMP"  : { "sy": getattr(model3, "sy", None),
        #                     "hk": getattr(model3, "hk", None) },
        #     "calib_HMP"   : getattr(model3, "crit_calib", None),
        #     "valid_HMP"   : getattr(model3, "crit_valid", None),
        #     "prev_HMP"    : crit_prev_HMP,
        #     "HMP_reseau_best": best_HMP_reseau,
        #     "params_HMP_reseau": { "sy": getattr(model4, "sy", None),
        #                         "hk": getattr(model4, "hk", None) },
        #     "calib_HMP_reseau": getattr(model4, "crit_calib", None),
        #     "valid_HMP_reseau": getattr(model4, "crit_valid", None),
        #     "prev_HMP_reseau": crit_prev_HMP_reseau,
        # }
        res2 = {
            "nom"         : nom,
            "station_id"  : id,
            "RL_best"     : best_RL2,
            "alpha_RL"    : getattr(model1, "alpha", None),
            "Vmax_RL"     : getattr(model1, "Vmax", None),
            "calib_RL"    : getattr(model1, "crit_calib", None),
            "valid_RL"    : getattr(model1, "crit_valid", None),
            "prev_RL"     : crit_prev_RL2,
            "GR4J_best"   : best_GR4J2,
            "params_GR4J" : getattr(model2, "x", None),
            "calib_GR4J"  : getattr(model2, "crit_calib", None),
            "valid_GR4J"  : getattr(model2, "crit_valid", None),
            "prev_GR4J"   : crit_prev_GR4J2,
            "HMP_reseau_best": best_HMP_reseau2,
            "params_HMP_reseau": { "sy": getattr(model4, "sy", None),
                                "hk": getattr(model4, "hk", None) },
            "calib_HMP_reseau": getattr(model4, "crit_calib", None),
            "valid_HMP_reseau": getattr(model4, "crit_valid", None),
            "prev_HMP_reseau": crit_prev_HMP_reseau2,
            "time_s"  : cpu_time_used
        }

    except Exception as e:
        print(f"Erreur station {nom} : {e}")

    print(f"[{proc} | PID {pid}] a fini de traiter le bv {nom}")

    return res2
    
def main():

    df_ref = pd.read_csv("ref_stations_autres.csv", sep=";")
    args_list = list(zip(df_ref["Name"], df_ref["id"], df_ref["x"], df_ref["y"], df_ref["Département"]))

    n_procs = 9
    print(f"Démarrage du pool avec {n_procs} processus...")

    with Pool(n_procs) as pool:
        # Répartition dynamique : chunksize=1 => chaque tâche est dispatchée indépendamment
        results = list(pool.imap_unordered(process_file, args_list, chunksize=1))

        # ancienne méthode chaque coeur obtient une liste à traiter
        #results = pool.map(process_file, args_list)  # results est une liste de tuples (res1, res2)

    # unzip
    # rows1 = [r[0] for r in results]
    # rows2 = [r[1] for r in results]

    rows = results

    # Création du DataFrame et export

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # path_avec = os.path.join(script_dir, f"Multi_mod_calibration_{fct_calib}_{transfo}_avecHMP.csv")
    # pd.DataFrame(rows1).to_csv(path_avec, index=False)

    path_sans = os.path.join(script_dir, f"Multi_mod_calibration_{fct_calib}_{transfo}_time.csv")
    pd.DataFrame(rows).to_csv(path_sans, index=False)

    print(f"Résumé des calibrations pour {fct_calib} et transfo {transfo} écrit dans {script_dir}")

if __name__ == "__main__":
    
    main()
