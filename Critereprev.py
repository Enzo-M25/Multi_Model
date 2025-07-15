  
import numpy as np
from scipy.optimize import minimize
from typing import Dict

class Critereprev :
    """
    Regroupe differentes fonctions permettant de calculer des criteres de performances pour le modele de reservoir lineaire

    Attributs
    Q_obs : Vecteur des débits mesurés sur une certaine période
    Q_sim : Vecteur des débits simulés sur une certaine période
    """

    def __init__(self, Q_obs:np.ndarray, Q_sim:np.ndarray):
        
        if Q_obs.shape != Q_sim.shape:
            raise ValueError("Q_obs et Q_sim doivent avoir la même longueur")

        self.Q_obs = Q_obs
        self.Q_sim = Q_sim

    def crit_NSE(self) -> float :
        """
        Calcule le critere NSE correspondant

        Paramètre de sortie :
        Valeur du NSE
        """

        Q_bar = np.mean(self.Q_obs)
        num = np.sum((self.Q_obs - self.Q_sim) ** 2)
        denom = np.sum((self.Q_obs - Q_bar) ** 2)
        return 1 - num / denom
    
    def crit_NSE_log(self) -> float :
        """
        Calcule le critère NSE-log
        
        Parametre de sortie :
        Valeur du NSE-log.
        """

        Q_bar = np.mean(self.Q_obs)
        eps = Q_bar/100
        obs = self.Q_obs + eps
        sim = self.Q_sim + eps
        log_obs = np.log(obs)
        log_sim = np.log(sim)

        num = np.sum((log_obs - log_sim) ** 2)
        den = np.sum((log_obs - np.mean(log_obs)) ** 2)
        return 1 - num / den
    
    def crit_RMSE(self) -> float:
        """
        Calcule le Root Mean Squared Error (RMSE) entre Q_obs et Q_sim.

        Returns
        -------
        Valeur du RMSE.
        """
        
        return np.sqrt(np.mean((self.Q_obs - self.Q_sim) ** 2))
    
    def crit_KGE(self) -> float:

        """
        Calcule l'indice Kling-Gupta Efficiency (KGE)

        Parametre de sortie
        Valeur du KGE.
        """
        
        mu_obs = self.Q_obs.mean()
        mu_sim = self.Q_sim.mean()

        sigma_obs = self.Q_obs.std(ddof=0)
        sigma_sim = self.Q_sim.std(ddof=0)

        r = np.corrcoef(self.Q_obs, self.Q_sim)[0, 1]

        alpha = sigma_sim / sigma_obs if sigma_obs != 0 else np.nan
        beta  = mu_sim / mu_obs       if mu_obs    != 0 else np.nan

        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    def crit_Biais(self) -> float:

        """
        Calcule le biais en pourcentage

        Parametre de sortie :
        Valeur du biais en % 
        """
        somme_obs = np.sum(self.Q_obs)
        if somme_obs == 0:
            return np.nan  # évite division par zéro
        return 100 * np.sum(self.Q_sim - self.Q_obs) / somme_obs
    
    def crit_mix(self,  weights: Dict[str, float], transfo: Dict[str, str]) -> float:
        """
        Calcule un melange pondere de differents criteres.

        Parametre d'entrees :
        weights : dictionnaire où les clés sont les noms des méthodes de critères (ex. 'crit_NSE', 'crit_RMSE') et les valeurs sont les poids correspondants
        transfo : dictionnaire où les clés sont les noms des méthodes de critères (ex. 'crit_NSE', 'crit_RMSE') et les valeurs sont les transformations appliquees aux debits (ie. "", "log", "inv")

        NB : les deux parametres sont supposés contenir le meme nobre d'elements

        Parametre de sortie :
            Valeur du critère mixte.
        """

        Q_obs_orig = self.Q_obs.copy()
        Q_sim_orig = self.Q_sim.copy()

        if set(weights.keys()) != set(transfo.keys()):
            raise KeyError("Les clés de 'weights' et de 'transfo' doivent être identiques.")


        # Recenser les méthodes de critères disponibles sous la forme d'un dictionnaire {name,self.crit_x}
        available = {
            name: getattr(self, name)
            for name in dir(self)
            if callable(getattr(self, name)) and name.startswith('crit_')
        }

        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("La somme des poids est nulle, impossible de normaliser")
        
        numerateur = 0.0

        for crit_name, poids in weights.items():

            self.Q_obs = Q_obs_orig.copy()
            self.Q_sim = Q_sim_orig.copy()

            t = transfo[crit_name].strip().lower()

            if t == "" :
                pass
            elif t == "log":
                if np.any(self.Q_obs <= 0) or np.any(self.Q_sim <= 0):
                    raise ValueError(f"Impossible d'appliquer 'log' sur des débits non positifs pour '{crit_name}'.")
                self.Q_obs = np.log(self.Q_obs)
                self.Q_sim = np.log(self.Q_sim)
            elif t == "inv":
                if np.any(self.Q_obs == 0) or np.any(self.Q_sim == 0):
                    raise ValueError(f"Impossible d'appliquer 'inv' (division par zéro) pour '{crit_name}'.")
                self.Q_obs = 1.0 / self.Q_obs
                self.Q_sim = 1.0 / self.Q_sim
            else:
                raise ValueError(f"Transformation inconnue '{t}' pour le critère '{crit_name}'.")

            valeur_crit = available[crit_name]()
            numerateur += valeur_crit * poids

            self.Q_obs = Q_obs_orig
            self.Q_sim = Q_sim_orig

        return numerateur / total_weight