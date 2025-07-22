  
import os
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.0'
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + r'C:\Program Files\R\R-4.5.0\bin'

from rpy2.robjects import r, globalenv

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import rpy2.robjects as ro
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import pandas2ri, IntVector, FloatVector, StrVector
#rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: None

from .Model import Model
from Jauge import Jauge

class GR4J(Model) :
    """
    Classe fille de Model
    Modele GR4J 
    #TODO Attention le package airGR a été modifié afin de prendre en compte de nouvelles fcts de calcul de critères

    Attributs
    t_calib (datetime) : période de calibration du modèle
    t_valid (datetime) : période de validation des débits
    t_prev (datetime) : période de prévision des débits
    transfo (list[str]) : liste contenant les transformations appliquees aux debits (ie. "", "log", "inv")
    fct_calib (str) : nom du critère sur lequel on effectue la calibration (NSE, NSE-log, KGE, RMSE, Biais)
    dict_crit ({str,float}) : (optionnel dans le cas d'un seul critere) dictionnaire des noms des criteres sur lesquels on effectue la calibration associes à leurs poids respectifs
    crit_calib (float) : meilleure valeur du critere de calibration obtenue lors de la calibration de celui-ci
    crit_valid (float) : valeur du critere de validation obtenue lors de la validation de celui-ci
    nom_model (str) : nom du modele (GR4J)
    x (list[float]) : liste des parametres du modele
    """

    def __init__(self, t_calib_start:str, t_calib_end:str, t_valid_start:str, t_valid_end:str, t_prev_start:str, t_prev_end:str,
                 transfo:list[str], fct_calib:str, dict_crit: Optional[dict[str, float]] = None) :
        
        super().__init__(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib, dict_crit)
        self.nom_model = "GR4J"
        self.x: float | None = None

    def idx_range(self, df:pd.DataFrame, start:datetime, end:datetime) -> list[int] :
        """
        Renvoie la liste des indices correspondant aux lignes de df["DatesR"] comprises entre deux dates données (incluses).

        Paramètre d’entrée :
        df : pandas.DataFrame, un DataFrame contenant une colonne 'DatesR' au format datetime.
        start : date de début de la période (ex. '2005-01-01').
        end : date de fin de la période (ex. '2010-12-31').

        Paramètre de sortie :
        liste des indices (entiers) des lignes dont la date est comprise entre start et end.
        """

        mask = (df["DatesR"].dt.date >= pd.to_datetime(start).date()) & \
            (df["DatesR"].dt.date <= pd.to_datetime(end).date())
        return [i+1 for i in df.index[mask]]

    def param_calib(self, bv:Jauge) -> None :
        """
        Permet de definir les attributs de classe crit_calib, x_1, x_2, x_3, x_4 et crit_valid suite à la calibration et validation du modèle sur le basin versant bv
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue la calibration
        """

        self.crit_calib, self.x = self.calibration(bv)
        self.crit_valid = self.validation(bv)

    def calibration(self, bv:Jauge) -> tuple[float,list[float]] :
        """
        Calibre le modele GR4J sur le bassin versant bv et recupere les parametres de calibration ainsi que le critere obtenus lors de celle-ci
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue la calibration

        Paramètres de sortie :
        crit_val : la meilleure valeur du critere obtenue apres la calibration
        x : liste des parametres du modele
        """

        #print("début calibration GR4J")

        bv.donnees["DatesR"] = pd.to_datetime(bv.donnees["tsd_date"].astype(str), format="%Y%m%d")

        # Préparer la conversion automatique pandas ↔ data.frame R
        pandas2ri.activate()

        # Exporter le DataFrame vers l’environnement global R
        ro.globalenv["data"] = bv.donnees

        ro.r('''
        library(airGR)
             
        InputsModel <- CreateInputsModel(
             FUN_MOD = RunModel_GR4J,
             DatesR = data$DatesR,
             Precip = data$tsd_prec,
             PotEvap = data$tsd_pet_ou
        )
        ''')

        Cal_Run = self.idx_range(bv.donnees, self.t_calib_start, self.t_calib_end)
        ro.globalenv["Cal_Run"] = IntVector(Cal_Run)

        r_transfo = StrVector(self.transfo)
        ro.globalenv["transfo"] = r_transfo

        if self.has_dict_crit() and self.fct_calib == "crit_mix":

            self.validate_weights()

            weights = FloatVector(list(self.dict_crit.values()))
            weights.names = StrVector(list(self.dict_crit.keys()))
            ro.globalenv["dict_crit"] = weights
            
            ro.r('''
                 
            crit_names <- names(dict_crit)
            fun_crit <- lapply(crit_names, function(critere_choisi) {
            switch(
                critere_choisi,
                crit_NSE      = ErrorCrit_NSE,
                crit_NSE_log  = ErrorCrit_NSE_log,
                crit_RMSE     = ErrorCrit_RMSE,
                crit_KGE      = ErrorCrit_KGE,
                crit_Biais    = ErrorCrit_Biais,
                stop("Critère inconnu : ", critere_choisi)
            )
            })

            weights <- as.list(as.numeric(dict_crit))

            n <- length(weights)
            obs <- rep(list(data$tsd_q_mm[Cal_Run]), n)
            tf <- transfo

            ''')

        elif not self.has_dict_crit() and not self.fct_calib == "crit_mix"  :

            if len(self.transfo) != 1 :
                raise ValueError(
                    "Incohérence entre le nombre de transformations et d'évaluations de critères demandés"
                )

            ro.globalenv["critere_choisi"] = StrVector([self.fct_calib])
            ro.r('''
            # Mapping du critère choisi
                    fun_crit <- switch(
                    critere_choisi,
                    crit_NSE = ErrorCrit_NSE,
                    crit_NSE_log = ErrorCrit_NSE_log,
                    crit_RMSE = ErrorCrit_RMSE,
                    crit_KGE = ErrorCrit_KGE,
                    crit_Biais = ErrorCrit_Biais,
                    stop("Critère inconnu : ", critere_choisi)
                    )
                 
            obs = data$tsd_q_mm[Cal_Run]
            tf = transfo
            weights = 1
                 
            ''')

        else :
            raise ValueError(f"Vous n'avez pas donne de critere ou les informations que vous avez rentrees sont incoherentes")
        
        ro.r('''
             
        CalRunOptions <- CreateRunOptions(
             FUN_MOD = RunModel_GR4J,
             InputsModel = InputsModel,
             IndPeriod_Run = Cal_Run
        )
             
        InputsCrit <- CreateInputsCrit(
             FUN_CRIT = fun_crit,
             InputsModel = InputsModel,
             RunOptions = CalRunOptions,
             Obs = obs,
             #VarObs = list("Q", "Q"),
             transfo = tf,
             Weights = weights
        )

        CalibOptions <- CreateCalibOptions(
             FUN_MOD = RunModel_GR4J,
             FUN_CALIB = Calibration_Michel
        )

        OutputsCalib <- Calibration(
             InputsModel = InputsModel,
             RunOptions = CalRunOptions,
             InputsCrit = InputsCrit,
             CalibOptions = CalibOptions,
             FUN_MOD = RunModel_GR4J,
             FUN_CALIB = Calibration_Michel
        )
        
        Param <- OutputsCalib$ParamFinalR
             
        ''')

        crit_v = ro.r("OutputsCalib$CritFinal")
        crit_val = float(crit_v[0])
        Param_r = ro.globalenv["Param"]

        #print("fin calibration GR4J")

        return crit_val, list(Param_r)

    def validation(self, bv:Jauge) -> float :
        """
        Effectue une validation des débits sur le bassin versant bv pour une certaine temporalité (t_valid)
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue l'estimation

        Paramètre de sortie :
        crit_val : la valeur de NSE obtenue apres l'estimation
        """

        #print("début validation GR4J")

        bv.donnees["DatesR"] = pd.to_datetime(bv.donnees["tsd_date"].astype(str), format="%Y%m%d")

        # Préparer la conversion automatique pandas ↔ data.frame R
        pandas2ri.activate()

        # Exporter le DataFrame vers l’environnement global R
        ro.globalenv["data"] = bv.donnees

        Sim_Run = self.idx_range(bv.donnees, self.t_valid_start, self.t_valid_end)
        ro.globalenv["Sim_Run"] = IntVector(Sim_Run)

        r_transfo = StrVector(self.transfo)
        ro.globalenv["transfo"] = r_transfo

        if self.has_dict_crit() and self.fct_calib == "crit_mix":

            self.validate_weights()

            weights = FloatVector(list(self.dict_crit.values()))
            weights.names = StrVector(list(self.dict_crit.keys()))
            ro.globalenv["dict_crit"] = weights
            
            ro.r('''
                 
            crit_names <- names(dict_crit)
            fun_crit <- lapply(crit_names, function(critere_choisi) {
            switch(
                critere_choisi,
                crit_NSE      = ErrorCrit_NSE,
                crit_NSE_log  = ErrorCrit_NSE_log,
                crit_RMSE     = ErrorCrit_RMSE,
                crit_KGE      = ErrorCrit_KGE,
                crit_Biais    = ErrorCrit_Biais,
                stop("Critère inconnu : ", critere_choisi)
            )
            })

            weights <- as.list(as.numeric(dict_crit))

            n <- length(weights)
            obs <- rep(list(data$tsd_q_mm[Sim_Run]), n)
            tf <- transfo

            ''')

        elif not self.has_dict_crit() and not self.fct_calib == "crit_mix" :

            if len(self.transfo) != 1 :
                raise ValueError(
                    "Incohérence entre le nombre de transformations et d'évaluations de critères demandés"
                )

            ro.globalenv["critere_choisi"] = StrVector([self.fct_calib])
            ro.r('''
            # Mapping du critère choisi
                    fun_crit <- switch(
                    critere_choisi,
                    crit_NSE = ErrorCrit_NSE,
                    crit_NSE_log = ErrorCrit_NSE_log,
                    crit_RMSE = ErrorCrit_RMSE,
                    crit_KGE = ErrorCrit_KGE,
                    crit_Biais = ErrorCrit_Biais,
                    stop("Critère inconnu : ", critere_choisi)
                    )
                 
            obs = data$tsd_q_mm[Sim_Run]
            tf = transfo
            weights = 1
                 
            ''')

        else :
            raise ValueError(f"Vous n'avez pas donne de critere ou les informations que vous avez rentrees sont incoherentes")

        ro.r('''
             
        SimRunOptions <- CreateRunOptions(
             FUN_MOD = RunModel_GR4J,
             InputsModel = InputsModel,
             IndPeriod_Run = Sim_Run
        )
        
        OutputsModel <- RunModel(
             InputsModel = InputsModel,
             RunOptions = SimRunOptions,
             Param = Param,
             FUN = RunModel_GR4J
        )
        
        InputsCritSim <- CreateInputsCrit(
             FUN_CRIT = fun_crit,
             InputsModel = InputsModel,
             RunOptions = SimRunOptions,
             Obs = obs,
             #VarObs =
             transfo = tf,
             Weights = weights
        )
            
        OutputsCritSim <- ErrorCrit(
             InputsCritSim,
             OutputsModel
        )
        ''')

        crit_v = ro.globalenv["OutputsCritSim"]
        crit_val = float(crit_v[0])

        #print("validation GR4J finie")

        return crit_val
    
    def prevision(self, bv:Jauge) -> tuple[pd.Series, np.ndarray] :
        """
        Effectue une prevision des débits sur le bassin versant bv pour une certaine temporalité (t_prev)
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue l'estimation

        Paramètres de sortie :
        d : temporalité de l'estimation sous forme de panda Series
        Q_sim : Vecteur des débits simulés pendant la période d sous forme de panda Series
        """

        #print("début prévision GR4J")

        Param = self.x

        bv.donnees["DatesR"] = pd.to_datetime(bv.donnees["tsd_date"].astype(str), format="%Y%m%d")

        # Préparer la conversion automatique pandas ↔ data.frame R
        pandas2ri.activate()

        # Exporter le DataFrame vers l’environnement global R
        ro.globalenv["data"] = bv.donnees

        Sim_Run = self.idx_range(bv.donnees, self.t_prev_start, self.t_prev_end)
        ro.globalenv["Sim_Run"] = IntVector(Sim_Run)

        ro.r('''
        library(airGR)
             
        SimRunOptions <- CreateRunOptions(FUN_MOD = RunModel_GR4J, InputsModel = InputsModel, IndPeriod_Run = Sim_Run)
        
        OutputsModel <- RunModel(InputsModel = InputsModel, RunOptions = SimRunOptions, Param = Param, FUN = RunModel_GR4J)
        ''')

        # Calculer l'efficacité Nash‐Sutcliffe sur la période de simulation et affichage résultats
        ro.r('''
        InputsCritSim <- CreateInputsCrit(FUN_CRIT = ErrorCrit_NSE, InputsModel = InputsModel, RunOptions = SimRunOptions, Obs = data$tsd_q_mm[Sim_Run])
            
        OutputsCritSim <- ErrorCrit_NSE(InputsCritSim, OutputsModel)
        ''')


        if Sim_Run:
            sim_indices_py = [i - 1 for i in Sim_Run]
            d = bv.donnees["DatesR"].iloc[sim_indices_py]
        else:
            d = pd.Series([], name="DatesR", dtype="datetime64[ns]")

        Q_sim = ro.globalenv["OutputsModel"].rx2("Qsim")

        #print("prévision GR4J finie")

        return d, Q_sim