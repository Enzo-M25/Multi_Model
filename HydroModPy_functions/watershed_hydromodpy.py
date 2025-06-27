# From HydroModPy

# Librairies

import argparse

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pkg_resources # Must be placed after DeprecationWarning as it is itself deprecated
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*declare_namespace.*')

# PYTHON PACKAGES
import sys
import os
import numpy as np
import pandas as pd
import flopy
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import imageio
import whitebox
wbt = whitebox.WhiteboxTools()
wbt.verbose = False

def pre_process_watershed(data) :

    # ROOT DIRECTORY

    from os.path import dirname, abspath
    root_dir = r"C:\USERS\enzma\Documents\HydroModPy"
    sys.path.append(root_dir)
    print("Root path directory is: {0}".format(root_dir.upper()))

    # HYDROMODPY MODULES

    #import src
    import importlib
    from src import watershed_root
    from src.watershed import climatic, geographic, geology, hydraulic, hydrography, hydrometry, intermittency, oceanic, piezometry, subbasin
    from src.display import visualization_watershed, visualization_results, export_vtuvtk
    from src.tools import toolbox, folder_root
    fontprop = toolbox.plot_params(8,15,18,20) # small, medium, interm, large

    #TODO MODIFIER TOUTE LA CELLULE POUR ADAPTER COM A TEST.PY
    #example_path = os.path.join(root_dir, 'Enzo')
    example_path = data[0]
    #data_path = os.path.join(example_path, 'data')
    data_path = data[1]

    #out_path = os.path.join(root_dir,'Enzo','results')
    out_path = data[2]

    print('The exemple directory is here :', example_path)
    print('The data comes from here :', data_path)
    print('The results of the example will be saved here :', out_path)



    # Name of the study site
    #TODO Nom et variables
    #watershed_name = 'Nancon' 
    #model_name = watershed_name
    watershed_name = data[3] 
    print('##### '+watershed_name.upper()+' #####')
    #first_year=2000
    #last_year=2021
    #time_step='D'

    #x = 389285.910
    #y = 6816518.749

    x = float(data[4])
    y = float(data[5])

    # Regional DEM
    #dem_path = os.path.join(data_path, 'regional dem.tif')
    dem_path = data[6]

    # Outlet coordinates of the catchment
    from_xyv = [x,y, 150, 10 , 'EPSG:2154']

    # Extract the catchment from a regional DEM
    BV = watershed_root.Watershed(dem_path=dem_path,
                                out_path=out_path,
                                load=False,
                                watershed_name=watershed_name,
                                from_lib=None, # os.path.join(root_dir,'watershed_library.csv')
                                from_dem=None, # [path, cell size]
                                from_shp=None, # [path, buffer size]
                                from_xyv=from_xyv, # [x, y, snap distance, buffer size]
                                bottom_path=None, # path 
                                save_object=True)



    # General plot of the study site
    visualization_watershed.watershed_local(dem_path, BV)

    # Clip specific data at the catchment scale
    BV.add_geology(data_path, types_obs='GEO1M.shp', fields_obs='CODE_LEG')
    BV.add_hydrography(data_path, types_obs=['regional stream network'])

    # Add hydrological data
    BV.add_hydrometry(data_path, 'france hydrometric stations.shp')
    BV.add_intermittency(data_path, 'regional onde stations.shp')

    # Extract a subbasin inside the study site
    BV.add_subbasin(os.path.join(data_path, 'additional'), 150)

    # Import modules
    BV.add_settings()
    BV.add_climatic()
    BV.add_hydraulic()

    # Visualization
    visualization_watershed.watershed_geology(BV)
    visualization_watershed.watershed_dem(BV)



    fig_dir = os.path.join(out_path, watershed_name, 'results_stable', '_figures')
    hydrometry_fig_dir = os.path.join(fig_dir, 'hydrometry')



    #TODO noms du csv
    #csv_name = 'hydrometry catchment Nancon.csv'
    csv_name = data[7]
    Qobs = pd.read_csv(data_path+'/'+csv_name, sep=';', index_col=0, parse_dates=True)
    Qobs = Qobs.squeeze()
    Qobs = Qobs.rename('Q')
    def select_period(df, first, last):
        df = df[(df.index.year>=first) & (df.index.year<=last)]
        return df
    area = BV.geographic.area

    #first = 1990
    #last = 2021 #TODO VARIABLES
    first = float(data[8])
    last = float(data[9])

    Qobs = select_period(Qobs, first, last)
    Qobs = (Qobs / (area*1000000)) * (3600 * 24) * 1000 # m3/s to mm/j
    data_index = Qobs.copy()

    mean_mensual = data_index.resample('M').mean() # mensual mean
    mean_annual = data_index.resample('Y').mean() # annual mean
    Mean = round(data_index.mean(),2)
    Mean = data_index.mean()
    Min = data_index.resample('Y').min()
    Q10 = data_index.resample('Y').quantile(0.10)
    Q25 = data_index.resample('Y').quantile(0.25)
    Q50 = data_index.resample('Y').quantile(0.50)
    Q75 = data_index.resample('Y').quantile(0.75)
    Q90 = data_index.resample('Y').quantile(0.90)
    Max = data_index.resample('Y').max()
    mean_interan_days = data_index.groupby([data_index.index.month,data_index.index.day], as_index=True).mean().to_frame()
    std_interan_days = data_index.groupby([data_index.index.month,data_index.index.day], as_index=True).std()
    q10_interan_days = data_index.groupby([data_index.index.month,data_index.index.day], as_index=True).quantile(0.10)
    q90_interan_days = data_index.groupby([data_index.index.month,data_index.index.day], as_index=True).quantile(0.90)
    q50_interan_days = data_index.groupby([data_index.index.month,data_index.index.day], as_index=True).quantile(0.50)
    mean_interan_days['std'] = std_interan_days
    mean_interan_days['q10'] = q10_interan_days
    mean_interan_days['q90'] = q90_interan_days
    mean_interan_days['q50'] = q50_interan_days
    mean_interan_days.index.names = ['months','days']
    mean_interan_days = mean_interan_days.reset_index()
    mean_interan_days = mean_interan_days.sort_values(['months','days'])
    mean_interan_days['counts'] = np.array(range(1,len(mean_interan_days)+1))

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(mean_interan_days.counts, mean_interan_days.q50, lw=2, color='darkred', label='Median')
    yerrmax = mean_interan_days.q90
    yerrmin = mean_interan_days.q10
    ax.fill_between(mean_interan_days.counts, yerrmin, yerrmax, color='cyan', edgecolor='grey', lw=0.5, alpha=0.5, label='10-90th')
    ax.set_yscale('log')
    ax.set_xlim(0,366)
    ax.set_ylim(0.01,10)
    ax.tick_params(axis='both', which='major', pad=10)
    x1 = np.linspace(0,366,13)
    squad = ['J','F','M','A','M','J','J','A','S','O','N','D','J']
    ax.set_xticks(x1)
    ax.set_xticklabels(squad, minor=False, rotation='horizontal')
    ax.set_xlabel('Months', labelpad=+10)
    ax.set_ylabel('Q / A [mm/d]',labelpad=+10)
    ax.set_title(watershed_name + ' [' + str(first) + ' to ' + str(last) + ']')
    ax.grid(alpha=0.25, zorder=0)

    #one = 2020 #TODO VARIABLE 
    one = float(data[10])

    dates = np.array([one],dtype=np.int64)
    colors = ['blue']
    for z in np.array(range(len(dates))):
        onlyone = data_index[(data_index.index.year==dates[z])].to_frame()
        onlyone = onlyone.groupby([onlyone.index.month, onlyone.index.day], as_index=True).mean()
        onlyone['counts'] = np.array(range(1,len(onlyone)+1))
        ax.plot(onlyone.counts, onlyone['Q'], color=colors[z], lw=1, label = str(dates[z]))
    ax.legend(loc='lower left')
    plt.tight_layout()
    filename = f"{watershed_name}_{first}_{last}.png"
    outpath = os.path.join(hydrometry_fig_dir, filename)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.close('all')

    # 1. Collecter récursivement tous les fichiers PNG
    all_png_files = []
    for root, dirs, files in os.walk(fig_dir):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                all_png_files.append(full_path)

    # 2. Afficher chaque image dans une fenêtre séparée
    for i, img_path in enumerate(all_png_files):
        try:
            # Créer une nouvelle figure avec un numéro explicite
            fig = plt.figure(i+1)
            
            # Charger et afficher l'image
            img = Image.open(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            
            # Afficher la fenêtre
            plt.show(block=True)
            plt.pause(0.1)  # Petite pause pour assurer le rendu
            
            # Attendre que l'utilisateur ferme la fenêtre
            while plt.fignum_exists(i+1):
                plt.pause(0.5)
            
        except Exception as e:
            print(f"Erreur avec {img_path}: {str(e)}")
            plt.close(i+1)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Fournit les data nécessaires au lancement de watershed"
    )

    parser.add_argument(
        "data",
        nargs="+",
        help="Liste de données str et float variables de watershed"
    )
    args = parser.parse_args()

    pre_process_watershed(args.data)