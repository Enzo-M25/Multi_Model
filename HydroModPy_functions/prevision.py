
# From HydroModPy

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
import os
from sys import platform
import geopandas as gpd
from datetime import datetime

# Libraries need to be installed if not
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import hydroeval as he

# # Libraries added from 'pip install' procedure
import deepdish as dd
import imageio
import hydroeval
import whitebox
wbt = whitebox.WhiteboxTools()
wbt.verbose = False
import xarray as xr
xr.set_options(keep_attrs = True)

from os.path import dirname, abspath
root_dir = r"C:\USERS\enzma\Documents\HydroModPy"
sys.path.append(root_dir)
print("Root path directory is: {0}".format(root_dir.upper()))

# Import HydroModPy modules
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

def prevision(nom_bv:str, first_year:int, last_year:int, freq_input:str, out_path:str, data_path:str, x:float, y:float, safransurfex:str, discharge_file:str, hk_ms:float, sy:float) -> None :

    """
    Effectue une prévision des débits en fonction des paramètres rentrés par l'utilisateur

    Paramètres d'entrée :
    nom_bv : nom du bassin versant
    first_year, last_year : années de début et de fin de calibration du modèle
    freq_input : pas de temps pour le modèle (journalier, mensuel)
    out_path : chemin du dossier où les résultats doivent être enregistrés
    data_path : chemin du dossier contenant les données  du bassin versant pour HydroModPy
    x, y : coordonnées de l'exutoire du bassin versant
    safransurfex : chemin du dossier contenant les données REA
    discharge_file : chemin du fichie csv contenant les données de débits observés
    hk_ms, sy : Paramètres optimaux du modèle retenus durant la phase de calibration
    """

    # PERSONAL PARAMETERS AND PATHS
    study_site = nom_bv
    first_year = first_year
    last_year = last_year
    freq_input = freq_input
    sim_state = 'transient' 
    parameters = "1.6e-5_2%"
    out_path = out_path
    data_path = data_path
    specific_data_path = os.path.join(data_path, study_site)

    print(f"out_path; {out_path}, Data path: {data_path}, specific_data_folder; {specific_data_path}")

    watershed_name = '_'.join([study_site])

    print('##### '+watershed_name.upper()+' #####')

    watershed_path = os.path.join(out_path, watershed_name)
    dem_path = os.path.join(data_path, 'regional dem.tif')

    load = True
    from_lib = None # os.path.join(root_dir,'watershed_library.csv')
    from_dem = None # [path, cell size]
    from_shp = None # [path, buffer size]
    from_xyv = [x, y, 150, 10 , 'EPSG:2154'] # [x, y, snap distance, buffer size, crs proj]
    bottom_path = None # path
    save_object = True

    # GEOGRAPHIC

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

    # DATA

    # visualization_watershed.watershed_local(dem_path, BV)

    # Clip specific data at the catchment scale
    # BV.add_geology(data_path, types_obs='GEO1M.shp', fields_obs='CODE_LEG')
    #BV.add_hydrography(data_path, types_obs=['regional stream network']) 
    BV.add_hydrometry(data_path, 'france hydrometric stations.shp')
    # BV.add_intermittency(data_path, 'regional onde stations.shp')
    # BV.add_piezometry()

    #Extract some subbasin from data available above
    # BV.add_subbasin(os.path.join(data_path, 'additional'), 150)

    # # General plot of the study site
    # visualization_watershed.watershed_geology(BV)
    # visualization_watershed.watershed_dem(BV)

    # climatic settings
    BV.add_climatic()
    first_year = first_year
    last_year = last_year

    # ##%%% Reanalyse
    # BV.climatic.update_sim2_reanalysis(var_list=['recharge', 'runoff', 'precip',
    #                                              'evt', 'etp', 't',
    #                                               ],
    #                                        nc_data_path=os.path.join(
    #                                            specific_data_path,
    #                                            r"Meteo\Historiques SIM2"),
    #                                        first_year=first_year,
    #                                        last_year=last_year,
    #                                        time_step=freq_input,
    #                                        sim_state=sim_state,
    #                                        spatial_mean=True,
    #                                        geographic=BV.geographic,
    #                                        disk_clip='watershed') # for clipping the netcdf files saved on disk

    # SAFRAN
    BV.add_safransurfex(safransurfex)

    # RECHARGE REANALYSIS
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

    # RUNOFF REANALYSIS
    BV.climatic.update_runoff_reanalysis(path_file=os.path.join(out_path, watershed_name, 'results_stable', 'climatic', '_RUN_D.csv'),
                                        clim_mod='REA',
                                        clim_sce='historic',
                                        first_year=first_year,
                                        last_year=last_year,
                                        time_step=freq_input,
                                        sim_state=sim_state)

    # BV.climatic.runoff = BV.climatic.runoff* BV.climatic.runoff.index.day #meandaypermonth to mm/month
    BV.climatic.update_runoff(BV.climatic.runoff / 1000, sim_state = sim_state) # from mm to m
    # BV.climatic.update_runoff(BV.climatic.runoff.resample('M').sum(), sim_state = sim_state)

    #% R and r ASSIGNATION
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

    # Qobs FORMATTING et F normalization 
    Qobs_path = os.path.join(data_path,discharge_file)
    Qobs = pd.read_csv(Qobs_path, delimiter=',')
    
    Qobs["Date (TU)"] = Qobs["Date (TU)"].str.split('T').str[0]
    Qobs["Date (TU)"] = pd.to_datetime(Qobs["Date (TU)"], format='%Y-%m-%d')
    Qobs.set_index("Date (TU)", inplace=True)

    Qobs = Qobs.drop(columns=["Statut", "Qualification", "Méthode", "Continuité"])
    Qobs = Qobs.squeeze()
    Qobs = Qobs.rename('Q')

    area = int(round(BV.geographic.area))
    Qobs = (Qobs / (area*1000000)) * (3600 * 24) # m3/s to m/day
    Qobsyear = Qobs.resample('Y').sum().mean() # m/day to m/y


    # R AND r RESAMPLE BY YEAR AND NORMALIZATION

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

    # DEFINE

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

    hk = hk_ms * 3600 * 24 # m/day utiliser best_hk ou best_hk_ms*3600*24 car modflow tourne au pas de tps jour
    cond_drain = None # or value of conductance
    sy = sy 

    # Boundary settings
    bc_left = None # or value
    bc_right = None # or value
    sea_level = 'None' # or value based on specific data : BV.oceanic.MSL
    split_temp = True

    iD_set_simulations = 'explorSy_test1'

    # UPDATE

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
    BV.hydraulic.update_sy(sy)

    # Boundary settings
    BV.settings.update_bc_sides(bc_left, bc_right)
    BV.add_oceanic(sea_level)
    BV.settings.update_dis_perlen(dis_perlen)

    # Particle tracking settings
    BV.settings.update_input_particles(zone_partic=BV.geographic.watershed_box_buff_dem) # or 'seepage_path'

    # MODFLOW

    # enlever les listes
    list_model_name = []
    list_success_modflow = []
    list_model_modflow = []

    model_name = iD_set_simulations+'_'+str(round(sy,4))+'_'+str(round(hk,3))+'_'+str(round(thick,3))
    BV.settings.update_model_name(model_name)
    print(model_name)

    model_modflow = BV.preprocessing_modflow(for_calib=False)
    success_modflow = BV.processing_modflow(model_modflow, write_model=True, run_model=True)

    list_model_name.append(model_name)
    list_success_modflow.append(success_modflow)
    list_model_modflow.append(model_modflow)

    dictio = {}
    dictio['list_model_name'] = list_model_name
    dictio['list_success_modflow'] = list_success_modflow
    dictio['list_model_modflow'] = list_model_modflow
    h5file = os.path.join(simulations_folder, 'results_listing_'+iD_set_simulations)
        
    dd.io.save(h5file, dictio)

    # RELOAD

    h5file = os.path.join(simulations_folder, 'results_listing_'+iD_set_simulations)
    d = dd.io.load(h5file)
    list_model_name = d['list_model_name'][:]
    list_success_modflow = d['list_success_modflow'][:]
    list_model_modflow = d['list_model_modflow'][:]

    # POSTPROCESSING

    for model_name, success_modflow, model_modflow in zip(list_model_name,
                                                        list_success_modflow,
                                                        list_model_modflow): # liste jusqu'ici
        if success_modflow == True:
            BV.postprocessing_modflow(model_modflow,
                                    watertable_elevation = True,
                                    watertable_depth= True, 
                                    seepage_areas = True,
                                    outflow_drain = True,
                                    groundwater_flux = True,
                                    groundwater_storage = True,
                                    accumulation_flux = True,
                                    persistency_index=True,
                                    intermittency_monthly=True,
                                    intermittency_daily=False,
                                    export_all_tif = False)

            timeseries_results = BV.postprocessing_timeseries(model_modflow=model_modflow,
                                                            model_modpath=None,
                                                            datetime_format=True, 
                                                            subbasin_results=True,
                                                            intermittency_monthly=True) # or None
            
            netcdf_results = BV.postprocessing_netcdf(model_modflow,
                                                    datetime_format=True)
            
    # CHRONICLE DISCHARGE PLOT

    simul_list = sorted(glob.glob(os.path.join(simulations_folder,
                                            iD_set_simulations+'*')),
                    key=os.path.getmtime)

    for i, simul in enumerate(simul_list[:]):

        model_name = os.path.split(simul)[-1]
            
        Smod_path = os.path.join(simul, 
                                r'_postprocess/_timeseries/_simulated_timeseries.csv')
        Smod = pd.read_csv(Smod_path, sep=';', index_col=0, parse_dates=True)
        
        Qmod = Smod['outflow_drain'] 
        Qmod = Qmod.squeeze()
        Qmod = Qmod*1000 # to mm

        if freq_input == 'M' :
            Qmod = (Qmod + (r * 1000)) * Qmod.index.day
        elif freq_input == 'W' :
            Qmod = (Qmod + (r * 1000)) * 7

        print (f'valeur de Qmod : {Qmod}')

        Qmod_stat = select_period(Qmod,first_year,last_year)

        prev_folder = os.path.join(watershed_path, "results_prevision")
        os.makedirs(prev_folder, exist_ok=True)

        df_mod = Qmod_stat.reset_index()
        df_mod.columns = ['date', 'Qmod']
        df_mod['date'] = pd.to_datetime(df_mod['date'])
        df_mod.to_csv(
            os.path.join(prev_folder, 'prevision_qmod.csv'),
            index=False
        )


if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(
        description="Fournit les data nécessaires au lancement de watershed"
    )

    parser.add_argument("nom_bv")
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year",  type=int)
    parser.add_argument("freq_input")
    parser.add_argument("out_path")
    parser.add_argument("data_path")
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("safransurfex")
    parser.add_argument("discharge_file")
    parser.add_argument("hk", type=float)
    parser.add_argument("sy", type=float)

    args = parser.parse_args()
    
    prevision(
        nom_bv         = args.nom_bv,
        first_year     = args.first_year,
        last_year      = args.last_year,
        freq_input     = args.freq_input,
        out_path       = args.out_path,
        data_path      = args.data_path,
        x              = args.x,
        y              = args.y,
        safransurfex   = args.safransurfex,
        discharge_file = args.discharge_file,
        hk_ms          = args.hk,
        sy             = args.sy,
    )