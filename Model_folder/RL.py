  
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime

from .Model import Model
from Jauge import Jauge
from CritereRL import CritereRL


class RL(Model) :
    """
    Classe fille de Model
    Modele de reservoir lineaire

    Attributs
    t_calib (datetime) : période de calibration du modèle
    t_valid (datetime) : période de validation des débits
    t_prev (datetime) : période de prévision des débits
    transfo (list[str]) : liste contenant les transformations appliquees aux debits (ie. "", "log", "inv")
    fct_calib (str) : nom du critère sur lequel on effectue la calibration (NSE, NSE-log, KGE, RMSE, Biais)
    dict_crit ({str,float}) : (optionnel dans le cas d'un seul critere) dictionnaire des noms des criteres sur lesquels on effectue la calibration associes à leurs poids respectifs
    crit_calib (float) : meilleure valeur du critere de calibration obtenue lors de la calibration de celui-ci
    crit_valid (float) : valeur du critere de validation obtenue lors de la validation de celui-ci
    nom_model (str) : nom du modele (RL | Reservoir lineaire)
    alpha (float) : parametre du modele, coefficient de "vidange" du bassin versant
    Vmax (float) : parametre du modele, capacite de stockage de l'aquifere
    """

    def __init__(self, t_calib_start:datetime, t_calib_end:datetime, t_valid_start:datetime, t_valid_end:datetime, t_prev_start:datetime, t_prev_end:datetime,
                 transfo:list[str], fct_calib:str, dict_crit: Optional[dict[str, float]] = None) :
        
        super().__init__(t_calib_start, t_calib_end, t_valid_start, t_valid_end, t_prev_start, t_prev_end, transfo, fct_calib, dict_crit)
        self.nom_model = "RL"
        self.alpha: float | None = None
        self.Vmax: float | None = None

    def idx_range(self, df:pd.Series, start:datetime, end:datetime) -> list[int]:
        """
        Renvoie la liste des indices correspondant aux lignes de df["DatesR"] comprises entre deux dates données (incluses).

        Paramètre d’entrée :
        df : pandas.DataFrame, un DataFrame contenant une colonne 'DatesR' au format datetime.
        start : date de début de la période (ex. '2005-01-01').
        end : date de fin de la période (ex. '2010-12-31').

        Paramètres de sortie :
        liste des indices (entiers) des lignes dont la date est comprise entre start et end.
        """

        mask = (df["DatesR"].dt.date >= pd.to_datetime(start).date()) & \
            (df["DatesR"].dt.date <= pd.to_datetime(end).date())
        return [i for i in df.index[mask]]

    def param_calib(self, bv:Jauge) -> None :
        """
        Permet de definir les attributs de classe crit_calib, alpha, Vmax et crit_valid suite à la calibration et la validation du modèle sur le basin versant bv
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue la calibration
        """

        self.crit_calib, self.alpha, self.Vmax = self.calibration(bv)
        self.crit_valid = self.validation(bv)

    def calibration(self, bv:Jauge) -> tuple[float,float,float] :
        """
        Effectue la calibration des paramètres du modèle de réservoir linéaire sur une certaine temporalité (t_calib)
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue la calibration

        Paramètres de sortie :
        crit_opt : Critère demandé par l'utilisateur lors de la phase de calibration
        alpha_opt, Vmax_opt : Paramètres optimaux du modèle de réservoir linéaire
        """

        #print("début calibration RL opti")

        alpha_0 = 0.0001
        Vmax_0 = 1

        # Extraction des donnees

        bv.donnees["DatesR"] = pd.to_datetime(bv.donnees["tsd_date"].astype(str), format="%Y%m%d")

        lignes = self.idx_range(bv.donnees, self.t_calib_start, self.t_calib_end)
        colonnes = ["DatesR", "tsd_q_mm", "tsd_prec", "tsd_pet_ou"]
        extrait = bv.donnees.loc[lignes, colonnes]
        d = extrait.iloc[:, 0].to_numpy()
        Q = extrait.iloc[:, 1].to_numpy()
        P = extrait.iloc[:, 2].to_numpy()
        E = extrait.iloc[:, 3].to_numpy()

        mask = np.isnan(Q) | np.isnan(P) | np.isnan(E)

        d = d[~mask]
        Q = Q[~mask]
        P = P[~mask]
        E = E[~mask]

        R = P-E 

        delta_t = 1.0

        c = CritereRL(Q, R, delta_t)
        if not self.has_dict_crit() and not self.fct_calib == "crit_mix"  :
            alpha_opt, Vmax_opt = c.optimize_criterion(alpha_0, Vmax_0, self.fct_calib, None, self.transfo)
            crit_opt, type_err = c.calculate_criteria(alpha_opt, Vmax_opt, self.fct_calib, None, self.transfo)
        else :
            self.validate_weights()
            alpha_opt, Vmax_opt = c.optimize_criterion(alpha_0, Vmax_0, self.fct_calib, self.dict_crit, self.transfo)
            crit_opt, type_err = c.calculate_criteria(alpha_opt, Vmax_opt, self.fct_calib, self.dict_crit, self.transfo)

        #print("calibration RL opti finie")

        return crit_opt, alpha_opt, Vmax_opt

    def validation(self, bv:Jauge) -> float :
        """
        Effectue une validation des débits sur le bassin versant bv pour une certaine temporalité (t_valid)
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue l'estimation

        Paramètre de sortie :
        crit_val : la valeur du critere obtenue apres la validation
        """

        #print("début validation RL")

        # Extraction des donnees

        bv.donnees["DatesR"] = pd.to_datetime(bv.donnees["tsd_date"].astype(str), format="%Y%m%d")

        lignes = self.idx_range(bv.donnees, self.t_valid_start, self.t_valid_end)
        colonnes = ["DatesR", "tsd_q_mm", "tsd_prec", "tsd_pet_ou"]
        extrait = bv.donnees.loc[lignes, colonnes]
        d = extrait.iloc[:, 0].to_numpy()
        Q = extrait.iloc[:, 1].to_numpy()
        P = extrait.iloc[:, 2].to_numpy()
        E = extrait.iloc[:, 3].to_numpy()

        mask = np.isnan(Q) | np.isnan(P) | np.isnan(E)

        d = d[~mask]
        Q = Q[~mask]
        P = P[~mask]
        E = E[~mask]

        R = P-E 

        delta_t = 1.0

        c = CritereRL(Q, R, delta_t)
        if not self.has_dict_crit() and not self.fct_calib == "crit_mix"  :
            crit_val, type_err = c.calculate_criteria(self.alpha, self.Vmax, self.fct_calib, None, self.transfo)
        else :
            self.validate_weights()
            crit_val, type_err = c.calculate_criteria(self.alpha, self.Vmax, self.fct_calib, self.dict_crit, self.transfo)

        #print("validation RL finie")

        return crit_val
    
    def prevision(self, bv:Jauge) -> tuple[pd.Series, np.ndarray] :
        """
        Effectue une prevision des debits sur le bassin versant bv pour une certaine temporalite (t_prev)
        
        Paramètre d’entrée :
        bv : Bassin versant jauge sur lequel on effectue l'estimation

        Paramètres de sortie :
        d : temporalité de l'estimation sous forme de panda Series
        Q_sim : Vecteur des débits simulés pendant la période d sous forme de panda Series
        """

        #print("début estimation RL")

        # Extraction des donnees

        bv.donnees["DatesR"] = pd.to_datetime(bv.donnees["tsd_date"].astype(str), format="%Y%m%d")

        lignes = self.idx_range(bv.donnees, self.t_prev_start, self.t_prev_end)
        colonnes = ["DatesR", "tsd_prec", "tsd_pet_ou"]
        extrait = bv.donnees.loc[lignes, colonnes]
        d = extrait.iloc[:, 0].to_numpy()
        P = extrait.iloc[:, 1].to_numpy()
        E = extrait.iloc[:, 2].to_numpy()

        mask = np.isnan(P) | np.isnan(E)

        d = d[~mask]
        P = P[~mask]
        E = E[~mask]

        R = P-E

        # Resolution avec les parametres de calibration

        v0 = self.Vmax/2
        delta_t = 1.0
        N = len(d)

        exp_alpha = np.exp(-self.alpha * delta_t)
        coeff_R   = (1 - exp_alpha) / self.alpha

        V = np.zeros(N)
        V[0] = v0

        for n in range(N - 1):
            V_pred = exp_alpha * V[n] + coeff_R * R[n]
            V[n+1] = min(max(V_pred, 0), self.Vmax)

        Q_sim = self.alpha * V

        d = pd.Series(d, name="DatesR")

        #print("estimation RL finie")

        return d, Q_sim

















    # def calibration_old(self, bv:Jauge) -> tuple[float,float,float] :
    #     """
    #     Calibre le modele de reservoir lineaire sur le bassin versant bv et recupere les parametres de calibration ainsi que le NSE obtenus lors de celle-ci
        
    #     Paramètre d’entrée :
    #     bv : Bassin versant jauge sur lequel on effectue la calibration

    #     Paramètres de sortie :
    #     crit_val : la meilleure valeur du critere de calibration obtenue
    #     a : valeur du parametre alpha pour bv apres calibration
    #     vmax : valeur du parametre Vmax pour bv apres calibration
    #     """

    #     print("début calibration RL")

    #     # Extraction des donnees

    #     bv.donnees["DatesR"] = pd.to_datetime(bv.donnees["tsd_date"].astype(str), format="%Y%m%d")

    #     lignes = self.idx_range(bv.donnees, self.t_calib_start, self.t_calib_end)
    #     colonnes = ["DatesR", "tsd_q_mm", "tsd_prec", "tsd_pet_ou"]
    #     extrait = bv.donnees.loc[lignes, colonnes]
    #     d = extrait.iloc[:, 0].to_numpy()
    #     Q = extrait.iloc[:, 1].to_numpy()
    #     P = extrait.iloc[:, 2].to_numpy()
    #     E = extrait.iloc[:, 3].to_numpy()

    #     mask = np.isnan(Q) | np.isnan(P) | np.isnan(E)

    #     d = d[~mask]
    #     Q = Q[~mask]
    #     P = P[~mask]
    #     E = E[~mask]

    #     R = P-E

    #     # Paramètres
    #     list_alpha = np.arange(0.0001, 0.5001, 0.005)
    #     #list_alpha = np.arange(0.0001, 0.5001, 0.01)
    #     list_Vmax  = np.arange(1, 550, 1)       

    #     delta_t = 1.0
    #     N = len(Q)

    #     # Pré‑allocation des résultats
    #     crit_mat   = np.zeros((len(list_alpha), len(list_Vmax)))

    #     # Boucles principales
    #     for i, alpha in enumerate(list_alpha):
    #         exp_alpha = np.exp(-alpha * delta_t)
    #         coeff_R   = (1 - exp_alpha) / alpha

    #         for j, Vm in enumerate(list_Vmax):
    #             V0 = Vm / 2.0

    #             # initialisation du vecteur V
    #             V = np.zeros(N)
    #             V[0] = V0

    #             # résolution
    #             for n in range(N - 1):
    #                 V_pred = exp_alpha * V[n] + coeff_R * R[n]
    #                 # on contraint V entre 0 et Vm
    #                 V[n+1] = min(max(V_pred, 0.0000001), Vm)

    #             # calcul du critere
    #             Q_sim   = alpha * V

    #             if self.has_dict_crit() and self.fct_calib == "crit_mix":

    #                 if len(self.transfo) != len(self.dict_crit):
    #                     raise ValueError(
    #                         f"Incohérence entre le nombre de transformations ({len(self.transfo)}) "
    #                         f"et le nombre de critères ({len(self.dict_crit)})."
    #                     )
    #                 self.validate_weights()

    #                 crit = CritereRL(Q,Q_sim)
    #                 tf = dict(zip(list(self.dict_crit.keys()), self.transfo))
    #                 crit_mat[i, j]  = crit.crit_mix(self.dict_crit, tf)

    #             elif not self.has_dict_crit() and not self.fct_calib == "crit_mix"  :
    #                 try:
    #                     if len(self.transfo) != 1 :
    #                         raise ValueError("Transformation de longueur inadaptée")
    #                     test = self.transfo[0]
    #                     if test == "" :
    #                         crit = CritereRL(Q,Q_sim)
    #                     elif test == "log" :
    #                         Q_bar = np.mean(Q)
    #                         eps = Q_bar/100
    #                         crit = CritereRL(np.log(Q.astype(float) + eps),np.log(Q_sim.astype(float) + eps))
    #                     elif test == "inv" :
    #                         crit = CritereRL(1/(Q.astype(float)),1/(Q_sim.astype(float)))
    #                     else :
    #                         raise ValueError(f"Transformation inconnue : {test}")

    #                     methode = getattr(crit, self.fct_calib)
    #                     crit_mat[i, j]  = methode()
    #                 except AttributeError:
    #                     raise ValueError(f"La fonction '{self.fct_calib}' n'existe pas dans Critere.")
                    
    #             else :
    #                 raise ValueError(f"Vous n'avez pas donne de critere ou les informations que vous avez rentrees sont incoherentes")

    #     # indices du tableau 
    #     ind_flat = crit_mat.argmax()
    #     ligne, col = np.unravel_index(ind_flat, crit_mat.shape)

    #     # conservation des meilleures valeurs
    #     crit_val = crit_mat[ligne, col]
    #     a = list_alpha[ligne]
    #     vmax = list_Vmax[col]

    #     print("calibration RL finie")

    #     return crit_val, a, vmax


    # def validation_old(self, bv:Jauge) -> float :
    #     """
    #     Effectue une validation des débits sur le bassin versant bv pour une certaine temporalité (t_valid)
        
    #     Paramètre d’entrée :
    #     bv : Bassin versant jauge sur lequel on effectue l'estimation

    #     Paramètre de sortie :
    #     crit_val : la valeur du critere obtenue apres la validation
    #     """

    #     print("début validation RL")

    #     # Extraction des donnees

    #     bv.donnees["DatesR"] = pd.to_datetime(bv.donnees["tsd_date"].astype(str), format="%Y%m%d")

    #     lignes = self.idx_range(bv.donnees, self.t_valid_start, self.t_valid_end)
    #     colonnes = ["DatesR", "tsd_q_mm", "tsd_prec", "tsd_pet_ou"]
    #     extrait = bv.donnees.loc[lignes, colonnes]
    #     d = extrait.iloc[:, 0].to_numpy()
    #     Q = extrait.iloc[:, 1].to_numpy()
    #     P = extrait.iloc[:, 2].to_numpy()
    #     E = extrait.iloc[:, 3].to_numpy()

    #     mask = np.isnan(Q) | np.isnan(P) | np.isnan(E)

    #     d = d[~mask]
    #     Q    = Q[~mask]
    #     P    = P[~mask]
    #     E    = E[~mask]

    #     R = P-E

    #     # Résolution avec les parametres de calibration

    #     v0 = self.Vmax/2
    #     delta_t = 1.0
    #     N = len(Q)

    #     exp_alpha = np.exp(-self.alpha * delta_t)
    #     coeff_R   = (1 - exp_alpha) / self.alpha

    #     V = np.zeros(N)
    #     V[0] = v0

    #     for n in range(N - 1):
    #         V_pred = exp_alpha * V[n] + coeff_R * R[n]
    #         V[n+1] = min(max(V_pred, 0.0000001), self.Vmax)

    #     # calcul du critere
    #     Q_sim   = self.alpha * V

    #     if self.has_dict_crit() and self.fct_calib == "crit_mix":

    #         if len(self.transfo) != len(self.dict_crit):
    #             raise ValueError(
    #                 f"Incohérence entre le nombre de transformations ({len(self.transfo)}) "
    #                 f"et le nombre de critères ({len(self.dict_crit)})."
    #             )
            
    #         self.validate_weights()

    #         crit = CritereRL(Q,Q_sim)
    #         tf = dict(zip(list(self.dict_crit.keys()), self.transfo))
    #         crit_val  = crit.crit_mix(self.dict_crit, tf)

    #     elif not self.has_dict_crit() and not self.fct_calib == "crit_mix"  :
                    
    #         try:
    #             if len(self.transfo) != 1 :
    #                 raise ValueError("Transformation de longueur inadaptée")
    #             test = self.transfo[0]
    #             if test == "" :
    #                 crit = CritereRL(Q,Q_sim)
    #             elif test == "log" :
    #                 Q_bar = np.mean(Q)
    #                 eps = Q_bar/100
    #                 crit = CritereRL(np.log(Q.astype(float) + eps),np.log(Q_sim.astype(float) + eps))
    #             elif test == "inv" :
    #                 crit = CritereRL(1/(Q.astype(float)),1/(Q_sim.astype(float)))
    #             else :
    #                 raise ValueError(f"Transformation inconnue : {test}")

    #             methode = getattr(crit, self.fct_calib)
    #             crit_val  = methode()
    #         except AttributeError:
    #             raise ValueError(f"La fonction '{self.fct_calib}' n'existe pas dans Critere.")
            
    #     else :
    #         raise ValueError(f"Vous n'avez pas donne de critere ou les informations que vous avez rentrees sont incoherentes")

    #     print("validation RL finie")

    #     return crit_val