  
import pandas as pd
import numpy as np

from Critereprev import Critereprev
from Jauge import Jauge
from Model_folder.RL import RL
from Model_folder.GR4J import GR4J
from Model_folder.HydroModPy import HydroModPy
from Choix import Choix
from Pre_process import Pre_Process
from Post_process import Outputs

import os
from os.path import dirname, abspath
from datetime import datetime, date

def critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit) -> float :
    crit = 0

    if fct_calib == "crit_mix" :

        total_weight = sum(list(dict_crit.values()))
        if total_weight == 0:
            raise ValueError("La somme des poids est nulle, impossible de normaliser")

        for i, crit_fct in enumerate(list(dict_crit.keys())) :
            elem = transfo[i]
            if elem == "":
                elem = None
                critere = Critereprev(Q_obs,Q_sim)
            elif elem == "log":
                Q_bar = np.mean(Q_obs)
                eps = 0 if model.name == "GR4J" else Q_bar / 100
                critere = Critereprev(np.log(Q_obs.astype(float) + eps),np.log(Q_sim.astype(float) + eps))
            elif elem == "inv" :
                critere = Critereprev(1/(Q_obs.astype(float)),1/(Q_sim.astype(float)))
            methode = getattr(critere, crit_fct)
            valeur = methode()
            crit += dict_crit.values()[i] * valeur
        crit = crit / total_weight

    else :
        elem = transfo[0]
        if elem == "":
            elem = None
            critere = Critereprev(Q_obs,Q_sim)
        elif elem == "log":
            Q_bar = np.mean(Q_obs)
            eps = 0*Q_bar/100
            critere = Critereprev(np.log(Q_obs.astype(float) + eps),np.log(Q_sim.astype(float) + eps))
        elif elem == "inv" :
            critere = Critereprev(1/(Q_obs.astype(float)),1/(Q_sim.astype(float)))
        methode = getattr(critere, fct_calib)
        crit = methode()
    
    return crit

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

    # J001401001 Nancon
    # 389358
    # 6816630

    # J721401001 Flume
    # 344966
    # 6797471

    # J708311001 Veuvre
    # 365833
    # 6796501

    # J014401001 Loisance
    # 372020
    # 6823398
    
    id = "J721401001" #########
    nom = "Flume_daily" ##########
    dossier = "C:\\Users\\enzma\\Documents\\rennes 1\\M2\\Semestre 2\\Stage\\codes_matlab_resev_lin\\stations"
    fichier = f"CAMELS_FR_tsd_{id}.csv"

    watershed = Pre_Process(
        #example_path=r"C:\Users\enzma\Documents\HydroModPy\Enzo",
        example_path=r"C:\Users\enzma\Documents\Tests_Modeles\Test_Multi_Modeles - Copie\Multi_model\HydroModPy_functions",
        data_path=r"C:\Users\enzma\Documents\HydroModPy\Enzo\data",
        results_path= r"C:\Users\enzma\Documents\HydroModPy\Enzo\results",
        basin_name=nom,
        x=344966, #########
        y=6797471, #########
        dem_raster=r"C:\Users\enzma\Documents\HydroModPy\Enzo\data\regional dem.tif",
        hydrometry_csv= f"{id}_QmnJ(n=1_non-glissant).csv",
        year_start=2000,
        year_end=2020,
        example_year=2010
    )

    bv = Jauge(id, nom, dossier, fichier, watershed)

    #watershed.pre_processing()

    fct_calib = "crit_NSE"

    transfo = [""]
    dict_crit = {"crit_KGE": 0.5, "crit_NSE": 0.5}

    t_calib_start = parse_date("2005-01-01")
    t_calib_end = parse_date("2010-12-31")
    t_valid_start = parse_date("2010-01-01")
    t_valid_end = parse_date("2020-12-31")
    t_prev_start = parse_date("2021-01-01") # pour hydromodpy l'année doit être complète pour les comparaisons
    t_prev_end = parse_date("2021-12-31")

    if t_calib_start > t_calib_end or t_valid_start > t_valid_end or t_prev_start > t_prev_end :
        raise ValueError(f"Format invalide pour une période, début et fin inversé")

    mac = Choix()
    
    # model1 = RL(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
    # model1.param_calib(bv)
    # print("\n=== Résultats du modèle de Résevoir linéaire (RL) ===")
    # print(f"\n résultats calculés avec le(s) critère(s) : {fct_calib} et une transformation : {transfo}")
    # print(f"  Alpha      : {model1.alpha}")
    # print(f"  Vmax       : {model1.Vmax}")
    # print(f"  {fct_calib} Calib  : {model1.crit_calib:.4f}")
    # print(f"  {fct_calib} Valid  : {model1.crit_valid:.4f}")
    # print("===============================\n")
    # mac.add_model(model1)
    
    # model2 = GR4J(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
    # model2.param_calib(bv)
    # print("\n=== Résultats du modèle GR4J ===")
    # print(f"\n résultats calculés avec le(s) critère(s) : {fct_calib} et une transformation : {transfo}")
    # print(f"{fct_calib} calibration : {model2.crit_calib:.4f}")
    # print(f"{fct_calib} validation : {model2.crit_valid:.4f}")
    # print("Paramètres calibrés :")
    # for i, val in enumerate(model2.x, start=1):
    #     print(f"  X{i} : {val}")
    # print("===============================\n")
    # mac.add_model(model2)
    
    
    model3 = HydroModPy(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib, r"C:\Users\enzma\Documents\Tests_Modeles\Test_Multi_Modeles - Copie\Multi_model\HydroModPy_functions",
                        'M', r"C:\Users\enzma\Documents\HydroModPy\Enzo\data\Meteo\REA", dict_crit=None)
    model3.param_calib(bv)
    print("\n=== Résultats du modèle HydroModPy ===")
    print(f"\n résultats calculés avec le(s) critère(s) : {fct_calib} et une transformation : {transfo}")
    print(f"{fct_calib} calibration : {model3.crit_calib:.4f}")
    print(f"{fct_calib} validation : {model3.crit_valid:.4f}")
    print("Paramètres calibrés :")
    print(f"  Sy      : {model3.sy}")
    print(f"  hk(m/s) : {model3.hk}")
    print("===============================\n")
    mac.add_model(model3)
    
    try :
        best = mac.comparaison_models(fct_calib) # best est une liste de model

        for i, model in enumerate(best):

            main_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(main_dir, f"figures_{nom}_{t_prev_start.year}_{t_prev_end.year}")
            os.makedirs(figures_dir, exist_ok=True)

            print(f"Prévision num {i+1} avec le modèle {model.nom_model} :\n")

            d, Q_sim = model.prevision(bv)

            result = Outputs(id,nom,figures_dir,d,Q_sim)
            result.affiche()

            if (len(Q_sim) == len(bv.serie_debit(t_prev_start,t_prev_end))) :
                Q_obs = bv.serie_debit(t_prev_start,t_prev_end)
            elif (len(Q_sim) == len(bv.serie_debit_mensuel(t_prev_start,t_prev_end))) :
                Q_obs = bv.serie_debit_mensuel(t_prev_start,t_prev_end)
            else :
                raise ValueError("Impossible d'afficher une comparaison simulé / observé. Pas assez de mesures de débits observées.")

            result_compar = Outputs(id,nom,figures_dir,d,Q_sim,Q_obs)
            result_compar.affiche()
            result_compar.affiche_nuage()

            # Critère de prévision

            print(f'{fct_calib} : {critere_prevision(model, Q_obs, Q_sim, fct_calib, transfo, dict_crit)}')

    except ValueError as e :
        print(f"Erreur lors de la sélection du modèle : {e}")
    
if __name__ == "__main__":
    main()


    # TODO Specific discharge