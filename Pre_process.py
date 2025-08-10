  
import subprocess
import os
import sys

from pathlib import Path
from typing import Optional

class Pre_Process:
    """
    Utilise la classe Watershed de HydroModPy pour afficher des informations sur le bassin versant choisi

    Attributs :
    example_path (str) : chemin du répertoire HydroModPy dans lequel le code tourne
    data_path (str) : chemin du répertoire dans lequel se trouve les données du bassin versant choisi
    results_path (str) : chemin du répertoire dans lequel enregistrer les résultats
    basin_name (str) : nom du bassin versant choisi
    departement (int) : numéro du département dans lequel est situé le bassin versant
    x,y (float) : coordonnées de l'exutoire du bassin versant
    dem_raster (str) : chemin du fichier contenant les données régionales
    hydrometry_csv (str) : fichier dans data_path contenant les données de mesure de débits du bassin versant
    year_start, year_end (int) : années de début et de fin pour une analyse informelle des débits
    example_year (int) : année choisi pour donner un exemple sur un plot
        
    env_root  (str) : chemin du répertoire où se trouve l'environnement hydromodpy-0.1
    python_exe (str) : exécutable python associé à l'environnement hydromodpy-0.1
    """

    def __init__(self, example_path: str, data_path: str, results_path: str, basin_name: str,
                 departement:int, x: float, y: float, dem_raster: str, hydrometry_csv: str,
                 year_start: int, year_end: int, example_year: int,
                 env_root: str = r"C:\ProgramData\anaconda3\envs\hydromodpy-0.1"
                ) :        
                
        # Validation des années
        if year_start >= year_end:
            raise ValueError("L'année de début doit être inférieure à l'année de fin")
        if not (year_start <= example_year <= year_end):
            raise ValueError("L'année d'exemple doit être comprise entre l'année de début et de fin")
        
        # Données membres
        self.example_path = example_path
        self.data_path = data_path
        self.results_path = results_path
        self.basin_name = basin_name
        self.departement = departement
        self.x = x
        self.y = y
        self.dem_raster = dem_raster
        self.hydrometry_csv = hydrometry_csv
        self.year_start = year_start
        self.year_end = year_end
        self.example_year = example_year
        
        # Environnement Conda
        self.env_root = env_root
        self.python_exe = os.path.join(env_root, "python.exe")
        
    def pre_processing(self) -> None :
        """
        Affiche des données informelles (géologie, situation géographique, réseau hydrographique, moyenne des débits sur x années, obsevations des étiages) sur le bassin versant
        à partir de fonctions HydroModPy situées dans les dossier de l'utilisateur
        """
        
        # donnees necessaire pour l'environnement hydromodpy-0.1
        env = os.environ.copy()
        
        gdal_bin = os.path.join(self.env_root, "Library", "bin")
        env["PATH"] = gdal_bin + os.pathsep + env.get("PATH", "")
        
        env["PROJ_LIB"] = os.path.join(self.env_root, "Library", "share", "proj")
        env["GDAL_DATA"] = os.path.join(self.env_root, "Library", "share", "gdal")
        
        env["PYTHONPATH"] = self.example_path + os.pathsep + env.get("PYTHONPATH", "")
        
        # Chemin du script à exécuter
        script_path = os.path.join(self.example_path, "watershed_hydromodpy.py")
        
        # Construction de la liste d'arguments str
        str_args = [
            self.example_path,
            self.data_path, 
            self.results_path,
            self.basin_name,
            str(self.departement),
            str(self.x),
            str(self.y),
            self.dem_raster,
            self.hydrometry_csv,
            str(self.year_start),
            str(self.year_end),
            str(self.example_year)
        ]
        
        cmd = [self.python_exe, script_path, *str_args]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            error_msg = f"Erreur d'exécution : {result.stderr}"
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg)
        # else:
        #     print(result.stdout)