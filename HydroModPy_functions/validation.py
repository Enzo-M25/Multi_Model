
# From HydroModPy

# LIBRAIRIES

# PYTHON

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
    df = df[(df.index.year>=first) & (df.index.year<=last)]
    return df

def crit_nselog(simulations, evaluation):
    """
    
    """

    simulations = 0.01 * simulations
    evaluation = 0.01 * evaluation

    nselog = 1 - (
            np.sum((np.log(evaluation) - np.log(simulations)) ** 2, axis=0, dtype=np.float64)
            / np.sum((np.log(evaluation) - np.mean(np.log(evaluation))) ** 2, dtype=np.float64)
    )

    return nselog

def evalution_criteria(calib) :
    """
    
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


def validation(nom_bv, first_year, last_year, freq_input, out_path, data_path, x, y, safransurfex, dicharge_file, hk_ms, sy, fct_calib, transfo, crit_list, weights_list) -> None :

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

    # OPTIONS
    # Name of the study site
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
    BV.add_hydrography(data_path, types_obs=['regional stream network']) 
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
    Qobs_path = os.path.join(data_path,dicharge_file)
    Qobs = pd.read_csv(Qobs_path, delimiter=',')
    #print (Qobs.columns)
    #print(Qobs.head())
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
    Qobsmonth = Qobs.resample('M').sum()
    Qobsweek = Qobs.resample('W').sum()
    Qobsweekmm = Qobsweek * 1000 # m/day to mm/week
    Qobsweekmm = select_period(Qobsweekmm, first_year, last_year)
    Qobsmonthmm = Qobsmonth * 1000 # m/day to mm/month
    Qobsmonthmm = select_period(Qobsmonthmm, first_year, last_year)
    Qobsday = Qobs.resample('D').sum()
    Qobsdaymm = Qobsday * 1000 # m/day to mm/week
    Qobsdaymm = select_period(Qobsdaymm, first_year, last_year)


    # R AND r RESAMPLE BY YEAR AND NORMALIZATION

    if freq_input == 'M':
        R = R*R.index.day
        r = r*r.index.day
    if freq_input == 'W':
        R = R*7 
        r = r*7
    Rannual = R.resample('Y').sum().mean()
    rannual = r.resample('Y').sum().mean()
    Qsafran = Rannual+rannual
    F = Qobsyear / Qsafran
    print (f'F = {F}')
    R = R * F
    r = r * F

    # DEFINE

    # Frame settings
    box = True # or False
    sink_fill = False # or True

    # sim_state = 'transient' # 'steady' or 'transient'
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

    # # Particle tracking settings
    # zone_partic = 'domain' # or watershed

    # plt.plot(hk/R)
    # plt.yscale('log')

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
        
        fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},
                                    figsize=(10,3))

        model_name = os.path.split(simul)[-1]
            
        Smod_path = os.path.join(simul, 
                                r'_postprocess/_timeseries/_simulated_timeseries.csv')
        Smod = pd.read_csv(Smod_path, sep=';', index_col=0, parse_dates=True)
        
        Qmod = Smod['outflow_drain'] 
        Qmod = Qmod.squeeze()
        Qmod = Qmod*1000 # to mm
        Qmod = (Qmod + (r * 1000)) 
        print (f'valeur de Qmod : {Qmod}')
        # Rmod = Smod['recharge'] 
        # print (f'valeur de Rmod : {Rmod}')
        
        yearsmaj = mdates.YearLocator(1)   # every year
        yearsmin = mdates.YearLocator(1)
        # monthsmaj = mdates.MonthLocator(6)  # every month
        # monthsmin = mdates.MonthLocator(3)
        # months_fmt = mdates.DateFormatter('%m') #b = name of month ?
        years_fmt = mdates.DateFormatter('%Y')

        ax = a0
        if freq_input == 'D':
            ax.plot(Qobsdaymm, color='k', lw=1, ls='-', zorder=0, label='observed')
        if freq_input == 'W':
            ax.plot(Qobsweekmm, color='k', lw=1, ls='-', zorder=0, label='observed')
        if freq_input == 'M':
            ax.plot(Qobsmonthmm, color='k', lw=1, ls='-', zorder=0, label='observed')
        ax.plot(Qmod, color='red', lw=1, label='modeled')
        ax.set_xlabel('Date')
        ax.set_ylabel('Q / A [mm/month]')
        ax.set_yscale('log')
        ax.set_ylim(0.0001, 1000)
        # years_5 = mdates.YearLocator(5)  # every 5 years
        # ax.xaxis.set_major_locator(years_5)
        ax.xaxis.set_minor_locator(yearsmin)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.set_xlim(pd.to_datetime(f'{first_year}-01'), pd.to_datetime(f'{last_year}-12'))
        # ax.set_xlim(pd.to_datetime(f'2023-01'), pd.to_datetime(f'2023-12'))
        ax.legend()
        ax.set_title(model_name.upper(), fontsize=10)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        # axb = ax.twinx()
        # axb.bar(Rmod.index, Rmod,color='blue', edgecolor='blue', lw=2.5)
        # axb.set_ylim(0,999)
        # axb.invert_yaxis()
        # axb.set_yticklabels([0.1,200])
        if freq_input == 'D':
            Qobs_stat = select_period(Qobsdaymm,first_year,last_year)
        if freq_input == 'W':
            Qobs_stat = select_period(Qobsweekmm,first_year,last_year)
        if freq_input == 'M':
            Qobs_stat = select_period(Qobsmonthmm,first_year,last_year)
            
        Qmod_stat = select_period(Qmod,first_year,last_year)


        crit_valid = 0

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
                crit_valid += tmp
            crit_valid = crit_valid / total_weight

        else :
            evaluation, type_err = evalution_criteria(fct_calib)
            elem = transfo[0]
            if elem == "":
                elem = None
            if evaluation == he.kge :
                crit_valid = he.evaluator(evaluation, Qmod_stat, Qobs_stat, transform=elem)[0][0]    
            else :
                crit_valid = he.evaluator(evaluation, Qmod_stat, Qobs_stat, transform=elem)[0]

        valid_folder = os.path.join(watershed_path, "results_valid")
        os.makedirs(valid_folder, exist_ok=True)

        df = pd.DataFrame({"crit_valid": [crit_valid]})
        output_file = os.path.join(valid_folder, "validation_result.csv")
        df.to_csv(output_file, index=False)

        df_obs = Qobs_stat.reset_index()
        df_obs.columns = ['date', 'Qobs']
        df_obs['date'] = pd.to_datetime(df_obs['date'])
        df_obs.to_csv(
            os.path.join(valid_folder, 'validation_qobs.csv'),
            index=False
        )

        df_mod = Qmod_stat.reset_index()
        df_mod.columns = ['date', 'Qmod']
        df_mod['date'] = pd.to_datetime(df_mod['date'])
        df_mod.to_csv(
            os.path.join(valid_folder, 'validation_qmod.csv'),
            index=False
        )

        
        NSE = he.evaluator(he.nse, Qmod_stat, Qobs_stat)[0]
        NSElog = he.evaluator(he.nse, Qmod_stat, Qobs_stat, transform='log')[0]
        RMSE = np.sqrt(np.nanmean((Qobs_stat.values-Qmod_stat.values)**2))
        KGE = he.evaluator(he.kge, Qmod_stat, Qobs_stat)[0][0]
        print(model_name.upper())
        print(f'NSE = {NSE}')
        print(f'NSElog = {NSElog}')
        print(f'RMSE = {RMSE}')
        print(f'KGE = {KGE}')
        
            # Store metrics in DataFrame
        metrics_df = pd.DataFrame({
            'model_name': [model_name],
            'NSE': [round(NSE, 2)],
            'NSElog': [round(NSElog, 2)],
            'RMSE': [round(RMSE, 2)],
            'KGE': [round(KGE, 2)]
        })
        
        # Define the CSV file path
        metrics_csv_path = os.path.join(simulations_folder, '_figures', 'model_metrics.csv')
        
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
        ax.set_xlim(0.1,1000)
        ax.set_ylim(0.1,1000)
        # ax.set_xlim(0.1,300)
        # ax.set_ylim(0.1,300)    
        ax.set_xlabel('$Q_{obs}$ / A [mm/month]', fontsize=12)
        ax.set_ylabel('$Q_{sim}$ / A [mm/month]', fontsize=12)
        fig.tight_layout()
                    
        fig.savefig(os.path.join(simulations_folder, '_figures',
                    'validation'+'.png'),
                    bbox_inches='tight')
    

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
    parser.add_argument("dicharge_file")
    parser.add_argument("hk", type=float)
    parser.add_argument("sy", type=float)
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
    
    validation(
        nom_bv       = args.nom_bv,
        first_year   = args.first_year,
        last_year    = args.last_year,
        freq_input   = args.freq_input,
        out_path     = args.out_path,
        data_path    = args.data_path,
        x            = args.x,
        y            = args.y,
        safransurfex = args.safransurfex,
        dicharge_file= args.dicharge_file,
        hk_ms        = args.hk,
        sy           = args.sy,
        fct_calib    = args.fct_calib,
        transfo      = args.transfo,
        crit_list    = crit_list,
        weights_list = weights_list
    )