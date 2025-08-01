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

def pre_process_watershed(example_path: str, data_path: str, results_path: str, basin_name: str, departement:int, x: float, y: float,
                          dem_raster: str, hydrometry_csv: str, year_start: int, year_end: int, example_year: int) -> None:
    
    """
    Affiche plusieurs graphiques, cartes, etc donnant des informations sur le basin versant à l'aide des fonctions d'HydroModPy
    (délimitations géographiques du bv, lithologie, réseau hydro, débits sur le bassin versant de year_start à year_end)

    Paramètres d'entrée :
    example_path (str) : chemin du dossier contenant les fonctions de HydroModPy
    data_path (str) : chemin du dossier contenant les données  du bassin versant pour HydroModPy
    results_path (str) : chemin du dossier où les résultats doivent être enregistrés
    basin_name (str) : nom du bassin versant
    x,y (float) : coordonnées de l'exutoire du bassin versant
    dem_raster (str) : chemin du fichier raster du bassin versant
    hydrometry_csv (str) : chemin du fichie csv contenant les données de débits observés
    year_start,year_end (int) : années de début et de fin pour lesquelles les affichages seront calculés
    example_year (int) : année comprise entre year_start et year_end pour laquelle on affiche explicitement les débits lors du preprocessing
    """


    example_path = example_path
    data_path = data_path
    out_path = results_path

    print('The exemple directory is here :', example_path)
    print('The data comes from here :', data_path)
    print('The results of the example will be saved here :', out_path)

    # Name of the study site
    
    watershed_name = basin_name
    print('##### '+watershed_name.upper()+' #####')

    # Regional DEM
    dem_path = dem_raster

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

    stream_network_path = os.path.join(data_path, 'regional_stream_network', f'{departement}')
    BV.add_hydrography(stream_network_path, types_obs=['regional stream network'])

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

    Qobs = pd.read_csv(data_path+'/'+hydrometry_csv, sep=',')
    Qobs["Date (TU)"] = Qobs["Date (TU)"].str.split('T').str[0]
    Qobs["Date (TU)"] = pd.to_datetime(Qobs["Date (TU)"], format='%Y-%m-%d')
    Qobs.set_index("Date (TU)", inplace=True)

    Qobs = Qobs.drop(columns=["Statut", "Qualification", "Méthode", "Continuité"])
    Qobs = Qobs.squeeze()
    Qobs = Qobs.rename('Q')
    area = BV.geographic.area

    first = year_start
    last = year_end

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

    one = example_year

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

    parser.add_argument("example_path")
    parser.add_argument("data_path")
    parser.add_argument("results_path")
    parser.add_argument("basin_name")
    parser.add_argument("departement", type=int)
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("dem_raster")
    parser.add_argument("hydrometry_csv")
    parser.add_argument("year_start", type=int)
    parser.add_argument("year_end", type=int)
    parser.add_argument("example_year", type=int)
    
    args = parser.parse_args()

    pre_process_watershed(
        example_path     = args.example_path,
        data_path        = args.data_path,
        results_path     = args.results_path,
        basin_name       = args.basin_name,
        departement      = args.departement,
        x                = args.x,
        y                = args.y,
        dem_raster       = args.dem_raster,
        hydrometry_csv   = args.hydrometry_csv,
        year_start       = args.year_start,
        year_end         = args.year_end,
        example_year     = args.example_year
    )
