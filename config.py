
import os

# Chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DOC = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))

# Chemins spécifiques pour HydroModPy
HYDROMODPY_FUNCTIONS = os.path.join(BASE_DIR, "HydroModPy_functions")
DATA_DIR = os.path.join(DOC, "HydroModPy", "Enzo", "data")
METEO_DIR = os.path.join(DATA_DIR, "Meteo", "REA")
RESULTS_DIR = os.path.join(DOC, "HydroModPy", "Enzo", "results")
DEM_FILE = os.path.join(DATA_DIR, "regional dem.tif")

# Chemins pour les stations
STATIONS_DIR = os.path.join(PROJECT_ROOT, "stations")

# Paramètres de configuration

PARAM_CONFIG = {
    "fct_calib" : "crit_NSE",
    "transfo" : ["log"],
    "dict_crit" : None, #{"crit_KGE": 0.5, "crit_NSE": 0.5}
    "t_calib_start" : "2005-01-01", 
    "t_calib_end" : "2010-12-31",
    "t_valid_start" : "2010-01-01",
    "t_valid_end" : "2020-12-31",
    "t_prev_start" : "2021-01-01",
    "t_prev_end" : "2021-12-31"
}

# Configuration du bassin versant

WATERSHED_CONFIG = {
    "nom": "Guillec",
    "id": "J302401001",
    "x": 179067,
    "y": 6858450,
    "num_dep": 29
}