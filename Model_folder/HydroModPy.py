
import subprocess
import os
import sys
import pandas as pd
from typing import Optional

from .Model import Model
from Pre_process import Pre_Process

# https://pypi.org/project/hydroeval/

class HydroModPy(Model):
    """
    Classe fille de Model
    Modele HydroModPy 

    Attributs
    t_calib : période de calibration du modèle
    t_valid : période de validation des débits
    t_prev : période de prévision des débits
    transfo : liste contenant les transformations appliquees aux debits (ie. "", "log", "inv")
    fct_calib : nom du critère sur lequel on effectue la calibration (NSE, NSE-log, KGE, RMSE, Biais)
    dict_crit : (optionnel dans le cas d'un seul critere) dictionnaire des noms des criteres sur lesquels on effectue la calibration associes à leurs poids respectifs
    crit_calib : meilleure valeur du critere de calibration obtenue lors de la calibration de celui-ci
    crit_valid : valeur du critere de validation obtenue lors de la validation de celui-ci
    nom_model : nom du modele (GR4J)
    sy : paramètre du modèle, coefficient de porosité
    hk : paramètre du modèle, conductivité hydraulique
    """

    def __init__(self, t_calib_start:str, t_calib_end:str, t_valid_start:str, t_valid_end:str, t_prev_start:str, t_prev_end:str,
                 transfo:list[str], fct_calib:str, example_path:str, env_root: str = r"C:\ProgramData\anaconda3\envs\hydromodpy-0.1",
                 dict_crit: Optional[dict[str, float]] = None) :
        
        super().__init__(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib, dict_crit)
        self.nom_model = "HydroModpy"
        self.sy: float | None = None
        self.hk: float | None = None

        self.example_path = example_path
        # Environnement Conda
        self.env_root = env_root
        self.python_exe = os.path.join(env_root, "python.exe")

    def param_calib(self, watershed:Pre_Process, freq_input, safransurfex) -> None :
        """
        Permet de definir les attributs de classe crit_calib, sy, hk et (crit_valid) suite à la calibration et validation du modèle sur le basin versant bv
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue la calibration
        """

        if self.has_dict_crit() and self.fct_calib == "crit_mix":

            if len(self.transfo) != len(self.dict_crit):
                raise ValueError(
                    f"Incohérence entre le nombre de transformations ({len(self.transfo)}) "
                    f"et le nombre de critères ({len(self.dict_crit)})."
                )
            self.validate_weights()

        try :
            self.calib(watershed.basin_name, watershed.year_start, watershed.year_end, freq_input, watershed.x, watershed.y, watershed.hydrometry_csv,
                    watershed.results_path, watershed.data_path, safransurfex)
        except ValueError as e :
            print(f"Erreur du fonctionnement d'HydroModPy : {e}")
            sys.exit(1)
        
        optim_results_path = f"{watershed.basin_name}_{watershed.year_start}_{watershed.year_end}\\results_calibration\\optimization_results\\optimization_results.csv"
        results = pd.read_csv(os.path.join(watershed.results_path,optim_results_path))
        self.crit_calib = results["best crit"].iloc[0]
        self.sy = results["best_sy"].iloc[0]
        self.hk = results["best_hk_ms"].iloc[0]

    def calib(self, nom_bv, first_year, last_year, freq_input, x, y, dicharge_file, out_path, data_path, safransurfex):
        """
        
        """
        
        # donnees necessaire pour l'environnement hydromodpy-0.1
        env = os.environ.copy()
        
        gdal_bin = os.path.join(self.env_root, "Library", "bin")
        env["PATH"] = gdal_bin + os.pathsep + env.get("PATH", "")
        
        env["PROJ_LIB"] = os.path.join(self.env_root, "Library", "share", "proj")
        env["GDAL_DATA"] = os.path.join(self.env_root, "Library", "share", "gdal")
        
        env["PYTHONPATH"] = self.example_path + os.pathsep + env.get("PYTHONPATH", "")
        
        # Chemin du script à exécuter
        script_path = os.path.join(self.example_path, "calib.py")
        
        # Construction de la liste d'arguments str
        str_args = [
            nom_bv,
            str(first_year),
            str(last_year),
            freq_input,
            str(x),
            str(y),
            dicharge_file,
            out_path,
            data_path,
            safransurfex,
            self.fct_calib
        ]

        cmd = [self.python_exe, script_path, *str_args, "--transfo",  *self.transfo]

        if self.has_dict_crit() and self.fct_calib == "crit_mix":
            crit_list    = list(self.dict_crit.keys())
            weights_list = [str(w) for w in self.dict_crit.values()]
            cmd += ["--crit_list", *crit_list, "--weights_list", *weights_list,]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            print("Erreur d'exécution :", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        else:
            print(result.stdout)