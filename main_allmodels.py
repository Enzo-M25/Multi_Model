  
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count, current_process

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

dossier = "C:\\Users\\enzma\\Documents\\rennes 1\\M2\\Semestre 2\\Stage\\codes_matlab_resev_lin\\stations"

fct_calib = "crit_NSE"

transfo = ["log"]
dict_crit = {"crit_KGE": 0.5, "crit_NSE": 0.5}

t_calib_start = parse_date("2005-01-01")
t_calib_end = parse_date("2010-12-31")
t_valid_start = parse_date("2010-01-01")
t_valid_end = parse_date("2020-12-31")
t_prev_start = parse_date("2021-01-01") # pour hydromodpy l'année doit être complète pour les comparaisons
t_prev_end = parse_date("2021-12-31")

def process_file(args):

    nom, id, x, y = args

    proc = current_process().name
    pid  = os.getpid()
    print(f"[{proc} | PID {pid}] traite le bv {nom}")

    # Valeurs par défaut si jamais un critère n'est pas défini
    crit_prev_RL = crit_prev_GR4J = crit_prev_HMP = None
    best_RL = best_GR4J = best_HMP = "No"

    try :
    
        fichier = f"CAMELS_FR_tsd_{id}.csv"

        watershed = Pre_Process(
            #example_path=r"C:\Users\enzma\Documents\HydroModPy\Enzo",
            example_path=r"C:\Users\enzma\Documents\Tests_Modeles\Test_Multi_Modeles - Copie\Multi_model\HydroModPy_functions",
            data_path=r"C:\Users\enzma\Documents\HydroModPy\Enzo\data",
            results_path= r"C:\Users\enzma\Documents\HydroModPy\Enzo\results",
            basin_name=nom,
            x=x,
            y=y,
            dem_raster=r"C:\Users\enzma\Documents\HydroModPy\Enzo\data\regional dem.tif",
            hydrometry_csv= f"{id}_QmnJ(n=1_non-glissant).csv",
            year_start=2000,
            year_end=2021,
            example_year=2010
        )

        bv = Jauge(id, nom, dossier, fichier, watershed)

        mac = Choix()
        
        model1 = RL(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
        model1.param_calib(bv)
        mac.add_model(model1)
        
        model2 = GR4J(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
        model2.param_calib(bv)
        mac.add_model(model2)
        
        
        model3 = HydroModPy(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib, r"C:\Users\enzma\Documents\Tests_Modeles\Test_Multi_Modeles - Copie\Multi_model\HydroModPy_functions",
                            'M', r"C:\Users\enzma\Documents\HydroModPy\Enzo\data\Meteo\REA", dict_crit=None)
        model3.param_calib(bv)
        mac.add_model(model3)

        best = mac.comparaison_models(fct_calib) # best est une liste de model

        for model in best:

            d, Q_sim = model.prevision(bv)

            if (len(Q_sim) == len(bv.serie_debit(t_prev_start,t_prev_end))) :
                Q_obs = bv.serie_debit(t_prev_start,t_prev_end)
            elif (len(Q_sim) == len(bv.serie_debit_mensuel(t_prev_start,t_prev_end))) :
                Q_obs = bv.serie_debit_mensuel(t_prev_start,t_prev_end)
            else :
                raise ValueError("Impossible d'afficher une comparaison simulé / observé. Pas assez de mesures de débits observées.")

            if model.nom_model == "RL" :
                best_RL = "Yes"
                crit_prev_RL = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)
            elif model.nom_model == "GR4J" :
                best_GR4J = "Yes"
                crit_prev_GR4J = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)
            elif model.nom_model == "HydroModpy" :
                best_HMP = "Yes"
                crit_prev_HMP = critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)

    except Exception as e:
        print(f"Erreur station {nom} : {e}")

    print(f"[{proc} | PID {pid}] a fini de traiter le bv {nom}")

    return {
        "nom"         : nom,
        "station_id"  : id,
        "RL_best"     : best_RL,
        "alpha_RL"    : getattr(model1, "alpha", None),
        "Vmax_RL"     : getattr(model1, "Vmax", None),
        "calib_RL"    : getattr(model1, "crit_calib", None),
        "valid_RL"    : getattr(model1, "crit_valid", None),
        "prev_RL"     : crit_prev_RL,
        "GR4J_best"   : best_GR4J,
        "params_GR4J" : getattr(model2, "x", None),
        "calib_GR4J"  : getattr(model2, "crit_calib", None),
        "valid_GR4J"  : getattr(model2, "crit_valid", None),
        "prev_GR4J"   : crit_prev_GR4J,
        "HMP_best"    : best_HMP,
        "params_HMP"  : { "sy": getattr(model3, "sy", None),
                          "hk": getattr(model3, "hk", None) },
        "calib_HMP"   : getattr(model3, "crit_calib", None),
        "valid_HMP"   : getattr(model3, "crit_valid", None),
        "prev_HMP"    : crit_prev_HMP,
    }
    
def main():
    
    df_ref = pd.read_csv("ref_stations_mini.csv", sep=";") #TODO
    args_list = list(zip(df_ref["Name"], df_ref["id"], df_ref["x"], df_ref["y"]))

    n_procs = min(cpu_count(), 10)
    print(f"Démarrage du pool avec {n_procs} processus...")

    start = time.time()
    with Pool(n_procs) as pool:
        rows = pool.map(process_file, args_list)
    elapsed = time.time() - start

    # Création du DataFrame et export
    df_res = pd.DataFrame(rows)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, f"Multi_mod_calibration_{fct_calib}_{transfo}_mini.csv")
    df_res.to_csv(summary_path, index=False)
    pd.DataFrame([{"elapsed_time_s": elapsed}]).to_csv(summary_path, mode="a", header=False, index=False)
    print(f"Résumé des calibrations pour {fct_calib} et transfo {transfo} écrit dans {summary_path}")
    print(f"⏱️ Temps parallèle : {elapsed:.2f}s")


if __name__ == "__main__":
    
    main()
