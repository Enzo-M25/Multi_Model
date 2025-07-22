
# From HydroModPy

# Librairies

# Filter warnings (before imports)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pkg_resources # Must be placed after DeprecationWarning as it is itself deprecated
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*declare_namespace.*')

# Libraries installed by default
import argparse
import sys
import glob
import logging
import os
import shutil
from typing import List, Callable, Any, Tuple, Dict
from PIL import Image
from sys import platform
import geopandas as gpd
from datetime import datetime
from scipy.optimize import minimize

# Libraries need to be installed if not
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# # Libraries added from 'pip install' procedure
import deepdish as dd
import imageio
import hydroeval as he
import whitebox
wbt = whitebox.WhiteboxTools()
wbt.verbose = False
import xarray as xr
xr.set_options(keep_attrs = True)

from os.path import dirname, abspath
root_dir = r"C:\\USERS\\enzma\\Documents\\HydroModPy"
sys.path.append(root_dir)
print("Root path directory is: {0}".format(root_dir.upper()))

# HydroModPy modules

from src import watershed_root
from src.watershed import climatic, geographic, geology, hydraulic, hydrography, hydrometry, intermittency, oceanic, piezometry, subbasin
from src.modeling import downslope, modflow, modpath, timeseries
from src.display import visualization_watershed, visualization_results, export_vtuvtk
from src.tools import toolbox, folder_root

fontprop = toolbox.plot_params(8,15,18,20) # small, medium, interm, large



def select_period(df, first, last):
    """
    Sélectionne les lignes d’un DataFrame en temps compris entre deux bornes

    Paramètres d'entrée :
    df : pandas.DataFrame contenant toutes les dates
    first : Année de début (incluse) de la période à extraire.
    last : Année de fin (incluse) de la période à extraire.

    Paramètre de sortie
    df : Sous-ensemble du df d'origine contenant uniquement les lignes dont l’année de l’index est comprise entre first et last
    """

    df = df[(df.index.year>=first) & (df.index.year<=last)]
    return df

def normalize(x:float, xmin:float, xmax:float) -> float :
    """
    Normalise une valeur dans l'intervalle [0, 1].

    Paramètres d'entrée :
    x : Valeur à normaliser
    xmin : Borne basse de l'échelle d'origine
    xmax : Bornes haute de l'échelle d'origine

    Paramètre de sortie :
    Valeur normalisée dans l'intervalle [0, 1]
    """

    return (x - xmin) / (xmax - xmin)

def denormalize(x_norm:float, xmin:float, xmax:float) -> float :
    """
    Rétablit la valeur originale à partir d’une valeur normalisée dans [0, 1].

    Paramètres d'entrée :
    x_norm : Valeur normalisée à dénormaliser (doit être dans [0, 1])
    xmin : Borne basse de l’échelle d’origine utilisée pour la normalisation
    xmax : Borne haute de l’échelle d’origine utilisée pour la normalisation

    Paramètre de sortie :
    Valeur dénormalisée
    """
    
    return x_norm * (xmax - xmin) + xmin

def filter_dates(dates:pd.DatetimeIndex, use_filter:Tuple[bool,bool], calib_dates:Tuple[str,str], season_start_end:Tuple[int,int,int,int]) -> pd.Series:
    """
    Filtre une série de dates selon des critères temporels et saisonniers

    Paramètres d'entrée :
    dates : Ensemble de dates à filtrer
    use_filter : (use_time_filter, use_seasonal_filter) :
        - use_time_filter : appliquer le filtre de période calée sur `calib_dates`
        - use_seasonal_filter : restreindre en plus à la saison définie
    calib_dates : (date_de_début, date_de_fin) de la période de calibration
    season_start_end : (mois_début, jour_début, mois_fin, jour_fin) de la saison à retenir

    Paramètre de sortie :
    pandas.Series de booléens, indexée par dates ; Vrai pour les dates conservées, False pour les dates exclues
    """

    use_time_filter = use_filter[0]
    use_seasonal_filter = use_filter[1]

    calib_start_date = calib_dates[0]
    calib_end_date =calib_dates[1]

    season_start_month = season_start_end[0]
    season_start_day = season_start_end[1]
    season_end_month = season_start_end[2]
    season_end_day = season_start_end[3]

    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    mask = pd.Series(True, index=dates)

    if use_time_filter:
        mask = mask & (dates >= calib_start_date) & (dates <= calib_end_date)

        if use_seasonal_filter:
            def is_in_season(date):
                start = pd.Timestamp(date.year, season_start_month, season_start_day)
                # Adjust end date if season ends in the next year
                if season_end_month < season_start_month:
                    end = pd.Timestamp(date.year + 1, season_end_month, season_end_day)
                else:
                    end = pd.Timestamp(date.year, season_end_month, season_end_day)
                
                return (date >= start) & (date <= end)
            
            seasonal_mask = dates.map(is_in_season)
            mask = mask & seasonal_mask

    return mask

def crit_nselog(sim:pd.Series, obs:pd.Series) -> float:
    """
    Calcule le NSE-log entre sim et obs

    Paramètres d'entrée :
    sim : débits simulés
    obs : débits observés
    
    Paramètre de sortie :
    nselog : Valeur du NSE-log.
    """

    epsilon = np.mean(obs)/100

    sim = epsilon + sim
    obs = epsilon + obs

    nselog = 1 - (
            np.sum((np.log(obs) - np.log(sim)) ** 2, axis=0, dtype=np.float64)
            / np.sum((np.log(obs) - np.mean(np.log(obs))) ** 2, dtype=np.float64)
    )

    return nselog

def evalution_criteria(calib:str) -> Tuple[Callable, int] :
    """
    Renvoie un tuple contenant les informations sur le critère demandé par l'utilisateur

    Paramètre d'entrée :
    calib : Critère de calibration demandé par l'utilisateur

    Paramètres de sortie :
    evaluation : Fonction de calcul correspondante au critère calib demandé
    type_err : Valeur dont on cherche à se rapprocher lorsque l'on calcule le critère calib 
    """

    if calib ==  "crit_NSE" :
        evaluation = he.nse
        type_err = 1
    elif calib == "crit_NSE_log" :
        evaluation = crit_nselog
        type_err = 1
    elif calib == "crit_RMSE" :
        evaluation = he.rmse
        type_err = 0
    elif calib == "crit_KGE" :
        evaluation = he.kge
        type_err = 1
    elif calib == "crit_Biais" :
        evaluation = he.pbias
        type_err = 0
    else :
        raise ValueError("Critère choisi non reconnu")
    
    return evaluation, type_err

def erreur_modele_norm(params_norm:Tuple[float,float], compt:int, sy_min_max:Tuple[float,float], loghk_min_max:Tuple[float,float], BV:Any,
                       thick:float, calibration_folder:str, use_filter:Tuple[bool,bool], calib_dates:Tuple[str,str], season_start_end:Tuple[int,int,int,int],
                       Qobsmm:pd.Series, freq_input:str, Qobsdaymm:pd.Series, Qobsweekmm:pd.Series, years:Tuple[int,int], r:pd.Series, Qobsmonthmm:pd.Series,
                       optimization_results:Dict[str,Any], all_Qmod_Qobs_results:List[Any], optim_folder:str, all_simulations_results:List[Dict[str,Any]],
                       transfo:List[str], fct_calib:str, crit_list:List[str], weights_list:List[float]):
    
    """
    Renvoie l'erreur commise lors du calcul du critère choisi par l'utilisateur avec les paramètres sélectionnés

    Paramètres d'entrée :
    params_norm : Paramètres hk et sy à tester pour le modèle
    compt : nombre de tentatives
    sy_min_max : Bornes dans lesquelles doit se trouver sy
    loghk_min_max : Bornes dans lesquelles doit se trouver log(hk)
    BV : instance de Watershed représentant le bassin versant
    thick : épaisseur sélectionnée pour le modèle
    calibration_folder : chemin du dossier dans lequel sont sauvegardés les résultats de calibration
    use_filter : (use_time_filter, use_seasonal_filter) :
        - use_time_filter : appliquer le filtre de période calée sur `calib_dates`
        - use_seasonal_filter : restreindre en plus à la saison définie
    calib_dates : (date_de_début, date_de_fin) de la période de calibration
    season_start_end : (mois_début, jour_début, mois_fin, jour_fin) de la saison à retenir
    Qobsmm, Qobsdaymm, Qobsweekmm, Qobsmonthmm : débits observés sur différentes périodes
    freq_input : pas de temps pour le modèle (journalier, mensuel)
    years : first_year et last_year pour la calibration
    r : Ruisselement
    optimization_results, all_Qmod_Qobs_results, all_simulations_results : fichiers contenant les résultats --- enregistrés dans optim_folder
    optim_folder : dossier contenant les résultats
    transfo : liste contenant les transformations appliquees aux debits (ie. "", "log", "inv")
    fct_calib : nom du critère sur lequel on effectue la calibration (NSE, NSE-log, KGE, RMSE, Biais)
    crit_list, weights_list : dans le cas où la calibration s'effectue sur plusieurs critères, ces listes contiennent le nom des critères et les poids associés

    Paramètre de sortie :
    error : Différence en valeur absolue entre la valeur calculée pour le critère et la valeur idéale devant être atteinte (ie. type_err de evaluation_criteria)
    """

    sy_min = sy_min_max[0]
    sy_max = sy_min_max[1]

    log_hk_min = loghk_min_max[0]
    log_hk_max = loghk_min_max[1]

    first_year = years[0]
    last_year = years[1]

    sy = denormalize(params_norm[1], sy_min, sy_max)
    log_hk = denormalize(params_norm[0], log_hk_min, log_hk_max)
    hk = 10**log_hk  # Convert log(K) to hk
    print(f'hk = {hk/24/3600}')

    # BV.hydraulic.update_thick(thick)
    BV.hydraulic.update_sy(sy)
    BV.hydraulic.update_hk(hk)

    timestamp = datetime.now().strftime("%H%M%S")
    model_name = f"optim_{compt}_{timestamp}_hk{hk/24/3600:.2e}_sy{sy*100:.2f}%_th{thick:.1f}"
    logging.info(f"\nSimulation {compt}: hk={hk/24/3600:.2e}m/s, sy={sy*100:.2f}%, thick={thick:.1f}m")
    BV.settings.update_model_name(model_name)
    BV.settings.update_check_model(plot_cross=False, check_grid=True)

    model_modflow = BV.preprocessing_modflow(for_calib=True)
    success_modflow = BV.processing_modflow(model_modflow, write_model=True, run_model=True)

    if not success_modflow:
        print("Échec de la simulation!")
        compt += 1 
        return 1e6

    BV.postprocessing_modflow(model_modflow,
                                watertable_elevation=True,
                                seepage_areas=True,
                                outflow_drain=True,
                                accumulation_flux=True,
                                watertable_depth=True,
                                groundwater_flux=False,
                                groundwater_storage=False,
                                intermittency_monthly=True,
                                export_all_tif=False)

    BV.postprocessing_timeseries(model_modflow, 
                                model_modpath=None, 
                                datetime_format=True)

    smod_path = os.path.join(calibration_folder, model_name, r'_postprocess/_timeseries/_simulated_timeseries.csv')
    
    if not os.path.exists(smod_path):
        
        print(f"Fichier de résultats non trouvé: {smod_path}")
        return 1e6

    sim_series = pd.read_csv(smod_path, sep=';', index_col=0, parse_dates=True)

    if "outflow_drain" not in sim_series.columns:
        print("Colonne 'outflow_drain' manquante dans le fichier de résultats!")
        print("Colonnes disponibles: %s", sim_series.columns.tolist())
        return 1e6

    simulated_series = sim_series['outflow_drain']
    if simulated_series.empty:
        print("Série temporelle vide!")
        return 1e6

    # ---- Filtrage des dates ----
    sim_dates = simulated_series.index
    date_mask = filter_dates(sim_dates, use_filter, calib_dates, season_start_end)
    filtered_dates = sim_dates[date_mask]
    simulated_Q = simulated_series[date_mask].values

    if len(filtered_dates) == 0:
        print("Aucune date ne correspond aux critères de filtrage!")
        return 1e6

    print(f"Utilisation de {len(filtered_dates)} dates sur {len(sim_dates)} pour la calibration")

    observed_Q = []

    print("Qobsmm index:", Qobsmm.index)
    print("filtered_dates:", filtered_dates)

    for date in filtered_dates:
        print(f"Processing date: {date}")
        # adapte la variable a entrer en fonction de freq_input
        if date in Qobsmm.index:
            observed_Q.append(Qobsmm.loc[date])        
            print(f"Found date: {date}, Q: {Qobsmm.loc[date]}")
        else:
            closest_date = Qobsmm.index[abs(Qobsmm.index - date).argmin()]
            observed_Q.append(Qobsmm.loc[closest_date])
            print(f"Closest date found: {closest_date}, Q: {Qobsmm.loc[closest_date, 'Q']}")

    print("observed_Q:", observed_Q)
    
    n = len(simulated_Q)
    if n == 0:
        return 1e6
        
    Smod = pd.read_csv(smod_path, sep=';', index_col=0, parse_dates=True)

    Qmod = Smod['outflow_drain'] 
    Qmod = Qmod.squeeze()
    Qmod = Qmod*1000
    
    if freq_input == 'M' :
        Qmod = (Qmod + (r * 1000)) * Qmod.index.day
    elif freq_input == 'W' :
        Qmod = (Qmod + (r * 1000)) * 7

    print(f"Qmod : {Qmod}")
 
    if freq_input == 'D':
        Qobs_stat = select_period(Qobsdaymm,first_year,last_year)
        print(f"Qobs : {Qobsdaymm}")

    if freq_input == 'W':
        Qobs_stat = select_period(Qobsweekmm,first_year,last_year)
        print(f"Qobs : {Qobsweekmm}")
        
    if freq_input == 'M':
        Qobs_stat = select_period(Qobsmonthmm,first_year,last_year)
        print(f"Qobs : {Qobsmonthmm}")
        
    Qmod_stat = select_period(Qmod,first_year,last_year)

    crit = 0

    if fct_calib == "crit_mix" :

        total_weight = sum(weights_list)
        if total_weight == 0:
            raise ValueError("La somme des poids est nulle, impossible de normaliser")

        for i, crit_fct in enumerate(crit_list) :
            evaluation, type_err = evalution_criteria(crit_fct) 
            elem = transfo[i]
            if elem == "":
                elem = None

            if evaluation == he.kge :
                tmp = weights_list[i] * he.evaluator(evaluation, Qmod_stat, Qobs_stat, transform=elem)[0][0]    
            else :
                tmp = weights_list[i] * he.evaluator(evaluation, Qmod_stat, Qobs_stat, transform=elem)[0]
            crit += tmp
        crit = crit / total_weight

    else :
        evaluation, type_err = evalution_criteria(fct_calib)
        elem = transfo[0]
        if elem == "":
            elem = None
        if evaluation == he.kge :
            crit = he.evaluator(evaluation, Qmod_stat, Qobs_stat, transform=elem)[0][0]    
        else :
            crit = he.evaluator(evaluation, Qmod_stat, Qobs_stat, transform=elem)[0]

    print(f'{fct_calib} : {crit}')

    error = (1 - crit) if type_err == 1 else crit

    if np.isnan(error) :
        return np.inf  # Large error if indicator is invalid

    current_simulation = {
    "iteration": compt,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": model_name,
    "hk": hk,
    "hk_ms": hk/24/3600,
    "log_hk": log_hk,  # Ajout du log(hk) dans les résultats
    "sy": sy,
    "thick": thick,
    f"{fct_calib}": crit,
    "error_crit": error,

    "filtered_points": len(filtered_dates),
    "total_points": len(sim_dates),
    }
    
    #all_Qmod_Qobs_results.append(Qmod_Qobs)
    pd.DataFrame(all_Qmod_Qobs_results).to_csv(os.path.join(optim_folder,'all_Qmod_Qosb_results.csv'))
    
    # Append the results to the list
    all_simulations_results.append(current_simulation)

    pd.DataFrame(all_simulations_results).to_csv(
        os.path.join(optim_folder, 'all_simulations_results.csv'), 
        index=False
    )

    # Save if the error is the best so far
    if error < optimization_results["best_error"]:
        optimization_results["model_name"] = model_name
        optimization_results["model_modflow"] = model_modflow
        optimization_results["best_error"] = error
        optimization_results["best_crit"] = crit
        optimization_results["best_params"] = {
            "hk": hk,
            "log_hk": log_hk,  # Ajout du log(hk) dans les meilleurs paramètres
            "sy": sy,
            "thick": thick
        }
        optimization_results["filtered_dates"] = filtered_dates
        optimization_results["Qobs"] = Qobs_stat
        optimization_results["Qmod"] = Qmod_stat
        logging.info("► Meilleure simulation jusqu'à présent ◄")

    compt += 1
    return error

def calibration(nom_bv: str, first_year: int, last_year: int, freq_input: str, x: float, y: float, dicharge_file: str,
                out_path: str, data_path: str, safransurfex: str, transfo:List[str], fct_calib: str, crit_list: List[str], weights_list: List[float]) -> None:
    
    """
    Effectue la calibration des paramètres du modèle HydroModPy

    Paramètres d'entrée :
    nom_bv : nom du bassin versant
    first_year, last_year : années de début et de fin de calibration du modèle
    freq_input : pas de temps pour le modèle (journalier, mensuel)
    x, y : coordonnées de l'exutoire du bassin versant
    dicharge_file : chemin du fichie csv contenant les données de débits observés
    out_path : chemin du dossier où les résultats doivent être enregistrés
    data_path : chemin du dossier contenant les données  du bassin versant pour HydroModPy
    safransurfex : chemin du dossier contenant les données REA
    transfo : liste contenant les transformations appliquees aux debits (ie. "", "log", "inv")
    fct_calib : nom du critère sur lequel on effectue la calibration (NSE, NSE-log, KGE, RMSE, Biais)
    crit_list, weights_list : dans le cas où la calibration s'effectue sur plusieurs critères, ces listes contiennent le nom des critères et les poids associés
    """

    # Personnal parameters and paths

    study_site = nom_bv
    first_year = first_year
    last_year = last_year
    freq_input = freq_input
    x = x
    y = y
    discharge_file = dicharge_file

    years = [first_year, last_year]
    sim_state = 'transient' 

    hk_init = 1.0e-5 * 3600 * 24 # m/day #* compris entre log_hk_min, log_hk_max = np.log10(1e-8*24*3600), np.log10(1e-2*24*3600)  # Log scale
    sy_init = 1 / 100 # 0.1% #* compris entre sy_min, sy_max = 0.1/100, 10/100
    out_path = out_path
    data_path = data_path
    discharge_path = os.path.join(data_path, discharge_file)
    specific_data_path = os.path.join(data_path, study_site)

    print(f"out_path; {out_path}, Data path: {data_path}, specific_data_folder; {specific_data_path}")

    watershed_name = '_'.join([study_site])

    print('##### '+watershed_name.upper()+' #####')

    watershed_path = os.path.join(out_path, watershed_name)
    dem_path = os.path.join(data_path, 'regional dem.tif')

    if os.path.isdir(watershed_path):
        shutil.rmtree(watershed_path)
        print(f"Ancien dossier supprimé : {watershed_path}")

    os.makedirs(watershed_path, exist_ok=True)
    print(f"Nouveau dossier créé : {watershed_path}")

    load = True
    from_lib = None # os.path.join(root_dir,'watershed_library.csv')
    from_dem = None # [path, cell size]
    from_shp = None # [path, buffer size]
    from_xyv = [x, y, 150, 10 , 'EPSG:2154'] # [x, y, snap distance, buffer size, crs proj]
    bottom_path = None # path
    save_object = True

    # Geographic

    BV = watershed_root.Watershed(dem_path=dem_path,
                              out_path=out_path,
                              load=load,
                              watershed_name=watershed_name,
                              from_lib=from_lib, # os.path.join(root_dir,'watershed_library.csv')
                              from_dem=from_dem, # [path, cell size]
                              from_shp=from_shp, # [path, buffer size]
                              from_xyv=from_xyv, # [x, y, snap distance, buffer size]
                              bottom_path=bottom_path, # path
                              save_object=save_object)

    # Paths generated automatically but necessary for plots
    stable_folder = os.path.join(out_path, watershed_name, 'results_stable')
    simulations_folder = os.path.join(out_path, watershed_name, 'results_simulations')
    calibration_folder = os.path.join(out_path, watershed_name, 'results_calibration')

    #BV.add_hydrography(data_path, types_obs=['regional stream network'])
    BV.add_hydrometry(data_path,'france hydrometric stations.shp')
    
    # Recharge et ruisselement de surface direct

    BV.add_climatic()

    # Reanalyse
    BV.climatic.update_sim2_reanalysis(var_list=[ 'recharge', 'runoff',
                                                  'precip',
                                                  'evt', 'etp', 't', 
                                                ],
                                        nc_data_path=os.path.join(
                                            data_path,
                                            f"Meteo\{watershed_name}\Historiques SIM2"),
                                        first_year=first_year,
                                        last_year=last_year,
                                        time_step=freq_input,
                                        sim_state=sim_state,
                                        spatial_mean=True,
                                        geographic=BV.geographic,
                                        disk_clip='watershed') 
                                                                    
    # # # # Units
    # BV.climatic.evt = BV.climatic.evt / 1000 # from mm to m
    # BV.climatic.etp = BV.climatic.etp / 1000 # from mm to m
    # BV.climatic.precip = BV.climatic.precip / 1000 # from mm to m
    # BV.climatic.t = BV.climatic.t / 1000 # from mm to m

    # Besoin de le mettre à jour qu'une fois par an.
    BV.add_safransurfex(safransurfex)

    # Recharge reanalysis

    BV.climatic.update_recharge_reanalysis(path_file=os.path.join(out_path, watershed_name, 'results_stable', 'climatic', '_REC_D.csv'),
                                       clim_mod='REA',
                                       clim_sce='historic',
                                       first_year=first_year,
                                       last_year=last_year,
                                       time_step=freq_input,
                                       sim_state=sim_state)

    # BV.climatic.recharge = BV.climatic.recharge * BV.climatic.recharge.index.day #meandaypermonth to mm/month
    BV.climatic.update_recharge(BV.climatic.recharge/1000, sim_state = sim_state) # from mm to m
    # BV.climatic.update_recharge(BV.climatic.recharge.resample('M').sum(), sim_state = sim_state) # days to month

    BV.climatic.update_runoff_reanalysis(path_file=os.path.join(out_path, watershed_name, 'results_stable', 'climatic', '_RUN_D.csv'),
                                        clim_mod='REA',
                                        clim_sce='historic',
                                        first_year=first_year,
                                        last_year=last_year,
                                        time_step=freq_input,
                                        sim_state=sim_state)

    BV.climatic.update_runoff(BV.climatic.runoff/1000, sim_state=sim_state) # from mm to m

    # Recharge and runoff assignations

    if isinstance(BV.climatic.recharge, float):
        print(f"Time-space daily average value for recharge = {BV.climatic.recharge} m")
        print(f"Time-space daily average value for runoff = {BV.climatic.runoff} m")
    else:
        if isinstance(BV.climatic.recharge, xr.core.dataset.Dataset):
            R = BV.climatic.recharge.drop('spatial_ref').mean(dim = ['x', 'y']).to_pandas().iloc[:,0]
            r = BV.climatic.runoff.drop('spatial_ref').mean(dim = ['x', 'y']).to_pandas().iloc[:,0]
        elif isinstance(BV.climatic.recharge, pd.core.series.Series):  
            R = BV.climatic.recharge
            r = BV.climatic.runoff

    # Qobs formatting and F normalisation

    Qobs_path = os.path.join(discharge_path)
    Qobs = pd.read_csv(Qobs_path, delimiter=',')

    # Split the values at 'T' for the 'Date(TU)' column and remove the values after 'T'
    Qobs["Date (TU)"] = Qobs["Date (TU)"].str.split('T').str[0]
    Qobs["Date (TU)"] = pd.to_datetime(Qobs["Date (TU)"], format='%Y-%m-%d')
    Qobs.set_index("Date (TU)", inplace=True)

    Qobs = Qobs.drop(columns=["Statut", "Qualification", "Méthode", "Continuité"])
    Qobs = Qobs.squeeze()
    Qobs = Qobs.rename('Q')

    area = int(round(BV.geographic.area))
    Qobs = (Qobs / (area*1000000)) * (3600 * 24) # m3/s to m/day
    Qobsyear = Qobs.resample('Y').sum().mean() # m/day to m/y

    # Q resample by timescale

    Qobsmonth = Qobs.resample('M').sum()
    Qobsweek = Qobs.resample('W').sum()
    Qobsweekmm = Qobsweek * 1000 # m/day to mm/week
    Qobsweekmm = select_period(Qobsweekmm, first_year, last_year)
    Qobsmonthmm = Qobsmonth * 1000 # m/day to mm/month
    Qobsmonthmm = select_period(Qobsmonthmm, first_year, last_year)
    Qobsday = Qobs.resample('D').sum()
    Qobsdaymm = Qobsday * 1000 # m/day to mm/week
    Qobsdaymm = select_period(Qobsdaymm, first_year, last_year)

    if freq_input == 'D':
        Qobsmm = select_period(Qobsdaymm,first_year,last_year)
        print(f"Qobs : {Qobsdaymm}")

    if freq_input == 'W':
        Qobsmm = select_period(Qobsweekmm,first_year,last_year)
        print(f"Qobs : {Qobsweekmm}")
                
    if freq_input == 'M':
        Qobsmm = select_period(Qobsmonthmm,first_year,last_year)
        print(f"Qobs : {Qobsmonthmm}")

    # Recharge and runoff resample by year and normalisation 

    groundwater = R
    surfacewater = r
    if freq_input == 'M':
        groundwater = R*R.index.day
        surfacewater = r*r.index.day
    if freq_input == 'W':
        groundwater = R*7 
        surfacewater = r*7
    groundwater_annual = groundwater.resample('Y').sum().mean()
    surfacewater_annual = surfacewater.resample('Y').sum().mean()
    Qsafran = groundwater_annual+surfacewater_annual
    F = Qobsyear / Qsafran
    print (f'F = {F}')
    R = R * F
    r = r * F
    
    # Define

    # Frame settings
    box = True # or False
    sink_fill = False # or True

    sim_state = sim_state # 'steady' or 'transient'
    plot_cross = False
    dis_perlen = True

    # Climatic settings
    first_clim = 'first' # or 'first or value
    freq_time = freq_input

    # Hydraulic settings
    nlay = 1
    lay_decay = 10 # 1 for no decay
    bottom = None # elevation in meters, None for constant auifer thickness, or 2D matrix
    thick = 30 # if bottom is None, aquifer thickness
    hk = hk_init
    cond_drain = None # or value of conductance

    sy = sy_init

    # Boundary settings
    bc_left = None # or value
    bc_right = None # or value
    sea_level = 'None' # or value based on specific data : BV.oceanic.MSL
    split_temp = True

    # Update

    # Import modules
    BV.add_settings()
    BV.add_climatic()
    BV.add_hydraulic()

    # Frame settings
    BV.settings.update_box_model(box)
    BV.settings.update_sink_fill(sink_fill)
    BV.settings.update_simulation_state(sim_state)
    BV.settings.update_check_model(plot_cross=plot_cross)

    # Climatic settings
    recharge = R.copy()
    BV.climatic.update_recharge(recharge, sim_state=sim_state)
    BV.climatic.update_first_clim(first_clim)

    runoff = r.copy()
    BV.climatic.update_runoff(runoff, sim_state=sim_state)
    BV.climatic.update_first_clim(first_clim)

    # Hydraulic settings
    BV.hydraulic.update_nlay(nlay) # 1
    BV.hydraulic.update_lay_decay(lay_decay) # 1
    BV.hydraulic.update_bottom(bottom) # None
    BV.hydraulic.update_thick(thick) # 30 / intervient pas si bottom != None
    BV.hydraulic.update_hk(hk)
    BV.hydraulic.update_cond_drain(cond_drain)

    # Boundary settings
    BV.settings.update_bc_sides(bc_left, bc_right)
    BV.add_oceanic(sea_level)
    BV.settings.update_dis_perlen(dis_perlen)

    # Particle tracking settings
    BV.settings.update_input_particles(zone_partic=BV.geographic.watershed_box_buff_dem) # or 'seepage_path'

    ### Calibration

    run_optimization = True

    if run_optimization:
        optim_folder = os.path.join(calibration_folder, 'optimization_results')
        os.makedirs(optim_folder, exist_ok=True)

        all_simulations_results = []
        all_Qmod_Qobs_results = []
        use_time_filter = True
        
        calib_start_date = f"{first_year}-01-01"
        #calib_start_date = f"2005-01-01"
        calib_end_date = f"{last_year}-12-31"
        #calib_end_date = f"2010-12-31"
        
        use_seasonal_filter = False # true = calib sur période suivante
        season_start_month = 7
        season_start_day = 1
        season_end_month = 12
        season_end_day = 31

        use_filter = [use_time_filter, use_seasonal_filter]
        calib_dates = [calib_start_date, calib_end_date]
        season_start_end = [season_start_month, season_start_day, season_end_month, season_end_day]

        optimization_results = {"model_name": None, "model_modflow": None, "best_error": np.inf}
            
        # Define bounds and normalization factors
        log_hk_min, log_hk_max = np.log10(1e-8*24*3600), np.log10(1e-2*24*3600)  # Log scale
        sy_min, sy_max = 0.1/100, 10/100
        compt = 0

        sy_min_max = [sy_min,sy_max]
        loghk_min_max = [log_hk_min, log_hk_max]

        # Run the optimization
        print("\n=== DÉMARRAGE DE L'OPTIMISATION SIMPLEX ===")
        start_time = datetime.now()
        print(f"Démarrage: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if use_time_filter:
            filter_info = f"Période de calibration: {calib_start_date} à {calib_end_date}"
            if use_seasonal_filter:
                filter_info += f", saison: {season_start_day}/{season_start_month} à {season_end_day}/{season_end_month}"
            print(filter_info)

        # Define the bounds for the parameters using log scale for hk
        hk_min_mday, hk_max_mday = 1e-8 * 24 * 3600, 1e-2 * 24 * 3600  # m/day
        log_K_min = np.log10(hk_min_mday)  # log10 de la valeur en m/jour
        log_K_max = np.log10(hk_max_mday)  # log10 de la valeur en m/jour

        # Initial values
        log_hk_init = np.log10(hk)  # log10 de la valeur en m/jour
        hk_init = hk
        sy_init = sy

        # Convertir hk_init en log(hk_init) pour la normalisation
        log_hk_init = np.log10(hk_init)

        # Normalize the initial values with hk in log scale
        x0_norm = [
            normalize(log_hk_init, log_hk_min, log_hk_max),
            normalize(sy_init, sy_min, sy_max),
        ]

        # Log des bornes et valeurs initiales pour vérification
        print(f"Valeur initiale hk: {hk_init:.2e} m/jour ({hk_init/24/3600:.2e} m/s), log(hk): {log_hk_init:.4f}")
        print(f"Bornes hk: [{hk_min_mday:.2e}, {hk_max_mday:.2e}] m/jour, log(hk): [{log_K_min:.4f}, {log_K_max:.4f}]")
        
        # Run the optimization using the Nelder-Mead method (Simplex)
        result = minimize(
            erreur_modele_norm, 
            x0_norm, 
            args = (compt, sy_min_max, loghk_min_max, BV, thick, calibration_folder, use_filter, calib_dates, season_start_end,
                    Qobsmm, freq_input, Qobsdaymm, Qobsweekmm, years, r, Qobsmonthmm, optimization_results, all_Qmod_Qobs_results, optim_folder, all_simulations_results,
                    transfo, fct_calib, crit_list, weights_list),
            method='Nelder-Mead',
            options={
                'xatol': 0.01,
                'fatol': 0.01,
                'maxiter': 30,
                'disp': True
            }
        )
        
        # Conversion des résultats optimaux du log(hk) vers hk
        best_log_hk = denormalize(result.x[0], log_K_min, log_K_max)
        best_hk = 10**best_log_hk

        best_sy = denormalize(result.x[1], sy_min, sy_max)
        # best_thick = denormalize(result.x[2], thick_min, thick_max)

        end_time = datetime.now()
        duration = end_time - start_time

        # Use the best parameters for the final run
        BV.hydraulic.update_hk(best_hk)
        BV.hydraulic.update_sy(best_sy)
        # BV.hydraulic.update_thick(best_thick)

        # Update the model name with the best parameters
        model_name = f"final_optimized_hk{best_hk/24/3600:.2e}_sy{best_sy:.4f}_th{thick:.1f}"
        BV.settings.update_model_name(model_name)

        # Save the optimization results
        optim_results = {
            "best_hk": best_hk,
            "best_hk_ms": best_hk/24/3600,
            "best_log_hk": best_log_hk,  # Ajout du log(hk) dans les résultats
            "best_sy": best_sy,
            "best crit": optimization_results.get('best_crit'),
            "iterations": result.nfev,
            "duration_seconds": duration.total_seconds(),
            "optimization_start": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "optimization_end": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "best_model": optimization_results['model_name'],
            "time_filter": {
                "enabled": use_time_filter,
                "global_start": calib_start_date,
                "global_end": calib_end_date,
                "seasonal_filter": use_seasonal_filter,
                "season_start": f"{season_start_day}/{season_start_month}",
                "season_end": f"{season_end_day}/{season_end_month}"
            }
        }

        # Create optimization results folder if it doesn't exist
        optim_df = pd.DataFrame([optim_results])
        optim_df.to_csv(os.path.join(optim_folder, f'optimization_results.csv'), index=False)

        # Qobs
        s_obs = optimization_results['Qobs']
        df_obs = s_obs.reset_index()
        df_obs.columns = ['date', 'Qobs']
        df_obs['date'] = pd.to_datetime(df_obs['date'])
        df_obs.to_csv(
            os.path.join(optim_folder, 'optimization_results_qobs.csv'),
            index=False
        )

        # Qmod
        s_obs = optimization_results['Qmod']
        df_obs = s_obs.reset_index()
        df_obs.columns = ['date', 'Qmod']
        df_obs['date'] = pd.to_datetime(df_obs['date'])
        df_obs.to_csv(
            os.path.join(optim_folder, 'optimization_results_qmod.csv'),
            index=False
        )

        print(f"Résultats d'optimisation sauvegardés dans {optim_folder}")
    else: 
        print("Optimisation désactivée, utilisation des paramètres définis manuellement.")


    ### Formating Qobs Gauged station csv

    best_model_path = os.path.join(optim_folder, 'optimization_results.csv')
    best_model_df = pd.read_csv(best_model_path)
    best_model_col = 'best_model'
    if best_model_col in best_model_df.columns:
        simul = best_model_df[best_model_col].iloc[0]
    else:
        raise ValueError(f"Column '{best_model_col}' not found in optimization results.")

    simul = os.path.join(calibration_folder,simul)


    ### plot discharge
    fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},
                                figsize=(10,3))
    
    Smod_path = os.path.join(simul, r"_postprocess\_timeseries\_simulated_timeseries.csv")
    Smod = pd.read_csv(Smod_path, sep=';', index_col=0, parse_dates=True)

    Qmod = Smod['outflow_drain'] 
    Qmod = Qmod.squeeze()
    Qmod = Qmod*1000 # m to mm

    if freq_input == 'M' :
        Qmod = (Qmod + (r * 1000)) * Qmod.index.day
    elif freq_input == 'W' :
        Qmod = (Qmod + (r * 1000)) * 7

    print (f'valeur de Qmod : {Qmod}')

    yearsmin = mdates.YearLocator(1)
    years_fmt = mdates.DateFormatter('%Y')

    ax = a0
    if freq_input == 'D':
        ax.plot(Qobsdaymm, color='k', lw=1, ls='-', zorder=0, label='observed')
        ax.plot(Qmod, color='red', lw=1, label='modeled')
    if freq_input == 'W':
        ax.plot(Qobsweekmm, color='k', lw=1, ls='-', zorder=0, label='observed')
        ax.plot(Qmod, color='red', lw=1, label='modeled')
    if freq_input == 'M':
        ax.plot(Qobsmonthmm, color='k', lw=1, ls='-', zorder=0, label='observed')
        ax.plot(Qmod, color='red', lw=1, label='modeled')

    ax.set_xlabel('Date')
    ax.set_ylabel('Q / A [mm/month]')
    ax.set_yscale('log')
    ax.set_ylim(0.0001, 1000)
    ax.xaxis.set_minor_locator(yearsmin)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.set_xlim(pd.to_datetime(f'{first_year}-01'), pd.to_datetime(f'{last_year}-12'))
    ax.legend()
    ax.set_title(model_name.upper(), fontsize=10)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        
    if freq_input == 'D':
        Qobs_stat = select_period(Qobsdaymm,first_year,last_year)
    if freq_input == 'W':
        Qobs_stat = select_period(Qobsweekmm,first_year,last_year)
    if freq_input == 'M':
        Qobs_stat = select_period(Qobsmonthmm,first_year,last_year)
        
    Qmod_stat = select_period(Qmod,first_year,last_year)
            
    NSE = he.evaluator(he.nse, Qmod_stat, Qobs_stat)[0]
    NSElog = he.evaluator(he.nse, Qmod_stat, Qobs_stat, transform='log')[0]
    RMSE = np.sqrt(np.nanmean((Qobs_stat.values-Qmod_stat.values)**2))
    KGE = he.evaluator(he.kge, Qmod_stat, Qobs_stat)[0][0]
    print(model_name.upper())
    print(f" NSE {NSE}")
    print(f" NSElog {NSElog}")
    print(f" RMSE {RMSE}")
    print(f" KGE {KGE}")

        # Store metrics in DataFrame
    metrics_df = pd.DataFrame({
        'model_name': [model_name],
        'NSE': [round(NSE, 2)],
        'NSElog': [round(NSElog, 2)],
        'RMSE': [round(RMSE, 2)],
        'KGE': [round(KGE, 2)]
    })

    # Define the CSV file path
    figures_folder = os.path.join(simulations_folder, '_figures')
    metrics_csv_path = os.path.join(simulations_folder, '_figures', 'model_metrics.csv')
    if os.path.exists(figures_folder) == False :
        os.makedirs(os.path.join(simulations_folder, '_figures'), exist_ok=True)
        pass
    # Check if the file already exists to determine whether to write headers
    if os.path.isfile(metrics_csv_path):
        metrics_df.to_csv(metrics_csv_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_csv_path, index=False)


    ax = a1
    ax.scatter(Qobs_stat, Qmod_stat,
                s=25, edgecolor='none', alpha=0.75, facecolor='forestgreen')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot((0.01,1000),(0.01,1000), color='grey', zorder=-1)
    ax.set_xlim(0.01,1000)
    ax.set_ylim(0.01,1000)
    ax.set_xlabel('$Q_{obs}$ / A [mm/month]', fontsize=12)
    ax.set_ylabel('$Q_{sim}$ / A [mm/month]', fontsize=12)
                
    fig.savefig(os.path.join(simulations_folder, '_figures',
                'calibration'+'.png'),
                bbox_inches='tight')
    

    ### PLOT QMOD_QOBS for all simulation

    # Load the data from the CSV file
    all_Qmod_Qobs_path = os.path.join(optim_folder, 'all_Qmod_Qosb_results.csv')
    all_Qmod_Qobs_data = pd.read_csv(all_Qmod_Qobs_path)

    # Iterate through each simulation and plot Qobs and Qmod
    for index, row in all_Qmod_Qobs_data.iterrows():
        Qmod = pd.Series(eval(row['Qmod']))
        Qobs = pd.Series(eval(row['Qobs']))

        plt.figure(figsize=(10, 6))
        plt.plot(Qmod.index, Qmod.values, label='Qmod', color='blue')
        plt.plot(Qobs.index, Qobs.values, label='Qobs', color='orange', linestyle='--')
        plt.title(f"Simulation {index + 1}: Qmod vs Qobs")
        plt.xlabel("Time")
        plt.ylabel("Flow (mm/month)")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(optim_folder, f"Qmod_Qobs_simulation_{index + 1}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Plot saved for simulation {index + 1} at {plot_path}")


    plt.close('all')


if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(
        description="Fournit les data nécessaires au lancement de watershed"
    )

    parser.add_argument("nom_bv")
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year",  type=int)
    parser.add_argument("freq_input")
    parser.add_argument("x",           type=float)
    parser.add_argument("y",           type=float)
    parser.add_argument("dicharge_file")
    parser.add_argument("out_path")
    parser.add_argument("data_path")
    parser.add_argument("safransurfex")
    parser.add_argument("fct_calib")

    parser.add_argument(
        "--transfo",
        nargs="+",
        required=True,
        help="Liste des transformations à appliquer"
    )

    parser.add_argument(
        "--crit_list",
        nargs="+",
        help="Liste des noms de critères (ex.: crit_NSE crit_KGE)",
        metavar="CRIT"
    )

    parser.add_argument(
        "--weights_list",
        nargs="+",
        type=float,
        help="Liste des poids associés aux critères",
        metavar="WEIGHT"
    )

    args = parser.parse_args()
    
    # Prépare un dict ou des listes vides si non fournies
    crit_list    = args.crit_list    or [] 
    weights_list = args.weights_list or []
    
    calibration(
        nom_bv       = args.nom_bv,
        first_year   = args.first_year,
        last_year    = args.last_year,
        freq_input   = args.freq_input,
        x            = args.x,
        y            = args.y,
        dicharge_file= args.dicharge_file,
        out_path     = args.out_path,
        data_path    = args.data_path,
        safransurfex = args.safransurfex,
        fct_calib    = args.fct_calib,
        transfo      = args.transfo,
        crit_list    = crit_list,
        weights_list = weights_list
    )