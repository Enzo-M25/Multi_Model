  
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from Jauge import Jauge

class Model :
    """
    Classe mere des differents modeles consideres

    Attributs
    t_calib : période de calibration du modele
    t_valid : période de validation des débits
    t_prev : période de prévision des débits
    transfo : liste contenant les transformations appliquees aux debits (ie. "", "log", "inv")
    fct_calib : nom du critère sur lequel on effectue la calibration (NSE, NSE-log, KGE, RMSE, Biais)
    dict_crit : (optionnel dans le cas d'un seul critere) dictionnaire des noms des criteres sur lesquels on effectue la calibration associes à leurs poids respectifs
    crit_calib : meilleure valeur du critere de calibration obtenue lors de la calibration de celui-ci
    crit_valid : valeur du critere de validation obtenue lors de la validation de celui-ci
    nom_model : nom du modele
    """

    def __init__(self, t_calib_start:str, t_calib_end:str, t_valid_start:str, t_valid_end:str, t_prev_start:str, t_prev_end:str,
                 transfo:list[str], fct_calib:str, dict_crit: Optional[dict[str, float]] = None) :
        
        self.t_calib_start = t_calib_start
        self.t_calib_end = t_calib_end
        self.t_valid_start = t_valid_start
        self.t_valid_end = t_valid_end
        self.t_prev_start = t_prev_start
        self.t_prev_end = t_prev_end
        self.transfo = transfo
        self.fct_calib = fct_calib
        self._dict_crit: Optional[dict[str, float]] = dict_crit
        self.crit_calib: float | None = None
        self.crit_valid: float | None = None
        self.nom_model: str | None = None

    @property
    def dict_crit(self) -> dict[str, float]: 
        """
        Accès sécurisé à dict_crit.  Lève une erreur si dict_crit n’a pas été défini.
        """
        if self._dict_crit is None:
            raise AttributeError(
                "dict_crit n'a pas été initialisé. "
            )
        return self._dict_crit

    def has_dict_crit(self) -> bool:
        """
        Vérifie si dict_crit est disponible
        """
        return self._dict_crit is not None

    def validate_weights(self) -> bool:
        """
        Vérifie que :
        1. Toutes les clés de dict_crit figurent dans la liste des critères autorisés.
        2. Aucune combinaison interdite de critères n'apparaît dans dict_crit.

        Parametre de sortie :
        booleen indiquant si dict_crit contient des criteres impossibles à melanger
        """

        if len(self.transfo) != len(self.dict_crit):
            raise ValueError(
            f"Incohérence entre le nombre de transformations ({len(self.transfo)}) "
            f"et le nombre de critères ({len(self.dict_crit)})."
            )
        
        forbidden = [
            ("crit_NSE",     "crit_RMSE"),
            ("crit_NSE",     "crit_Biais"),
            ("crit_NSE_log", "crit_RMSE"),
            ("crit_NSE_log", "crit_Biais"),
            ("crit_RMSE",    "crit_KGE"),
            ("crit_RMSE",    "crit_Biais"),
            ("crit_KGE",     "crit_Biais"),
        ]

        available = {
            "crit_NSE",
            "crit_NSE_log",
            "crit_RMSE",
            "crit_KGE",
            "crit_Biais"
        }

        for crit_name in self.dict_crit:
            if crit_name not in available:
                raise ValueError(f"Critère inconnu : {crit_name}")

        for combo in forbidden:
            if all(c in self.dict_crit for c in combo):
                raise ValueError(f"Combinaison interdite de critères détectée : {combo}")

        return True

    @abstractmethod
    def idx_range(self, df, start, end):
        pass

    @abstractmethod
    def param_calib(self, bv:Jauge) -> None :
        pass

    @abstractmethod
    def calibration(self, bv:Jauge) -> tuple[float,float,float] :
        pass

    @abstractmethod
    def validation(self, bv:Jauge) -> float :
        pass

    @abstractmethod
    def prevision(self, bv:Jauge) -> tuple[pd.Series, pd.Series] :
        pass

    # @abstractmethod
    # def estimation_pediction(self, bv:Jauge) -> tuple[float, pd.Series, pd.Series] :