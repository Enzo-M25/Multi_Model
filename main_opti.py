
import pandas as pd

from Jauge import Jauge
from Model_folder.RL import RL
from Model_folder.GR4J import GR4J
from Choix import Choix
from Post_process import Outputs

import os
from os.path import dirname, abspath




# TODO Actuellement le RMSE




def main():
    
    id = "J260301001"
    nom = "Test"
    dossier = "C:\\Users\\enzma\\Documents\\rennes 1\\M2\\Semestre 2\\Stage\\codes_matlab_resev_lin\\stations"
    fichier = "CAMELS_FR_tsd_J260301001.csv"
    bv = Jauge(id, nom, dossier, fichier, None)

    fct_calib = "crit_KGE_opti"

    transfo = [""]
    dict_crit = {"crit_NSE": 0.5,"crit_KGE": 0.5}

    t_calib_start = "2005-01-01"
    t_calib_end = "2005-01-10"
    t_calib_end = "2010-12-31"
    t_valid_start = "2010-01-01"
    t_valid_end = "2020-12-31"
    t_prev_start = "2021-01-01"
    t_prev_end = "2021-12-31"

    mac = Choix()
    
    model1 = RL(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib)
    model1.param_calib_opti(bv)
    print("\n=== Résultats du modèle de Résevoir linéaire (RL) ===")
    print(f"  Alpha      : {model1.alpha}")
    print(f"  Vmax       : {model1.Vmax}")
    print(f"  {fct_calib} Calib  : {model1.crit_calib:.4f}")
    #print(f"  {fct_calib} Valid  : {model1.crit_valid:.4f}")
    print("===============================\n")
    mac.add_model(model1)

    try :
        best = mac.comparaison_models(fct_calib)

        d, Q_sim = best.prevision(bv)

        result = Outputs(id,nom,d,Q_sim)
        result.affiche()

        result_compar = Outputs(id,nom,d,Q_sim,bv.serie_debit(t_prev_start,t_prev_end))
        result_compar.affiche()

    except ValueError as e :
        print(f"Erreur lors de la sélection du modèle : {e}")
    
if __name__ == "__main__":
    main()