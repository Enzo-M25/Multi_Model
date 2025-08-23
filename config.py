
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
    "nom": "Flume",
    "id": "J721401001",
    "x": 344949,
    "y": 6797491,
    "num_dep": 35
}

# WATERSHED_CONFIG = {
#     "nom": "Arguenon",
#     "id": "J110301001",
#     "x": 305570,
#     "y": 6824711,
#     "num_dep": 22
# }

# WATERSHED_CONFIG = {
#     "nom": "Goyen",
#     "id": "J401401001",
#     "x": 143849,
#     "y": 6797335,
#     "num_dep": 29
# }

# WATERSHED_CONFIG = {
#     "nom": "Canut-Sud",
#     "id": "J797301001",
#     "x": 327336,
#     "y": 6753517,
#     "num_dep": 35
# }