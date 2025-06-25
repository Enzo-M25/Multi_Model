  
import numpy as np
from scipy.optimize import minimize
from typing import Dict

class CritereRL :
    """
    Regroupe differentes fonctions permettant de calculer des criteres de performances pour le modele de reservoir lineaire

    Attributs
    Q_obs : Vecteur des débits mesurés sur une certaine période
    Q_sim : Vecteur des débits simulés sur une certaine période
    """

    def __init__(self, Q_obs:np.ndarray, Q_sim:np.ndarray):
        
        if Q_obs.shape != Q_sim.shape:
            raise ValueError("Q_obs et Q_sim doivent avoir la même longueur")

        self.Q_obs = Q_obs.astype(float)
        self.Q_sim = Q_sim.astype(float)

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






    
    def _simulate_reservoir(self, alpha: float, Vmax: float, R: np.ndarray, delta_t: float) -> np.ndarray:
        """Simule le modèle du réservoir avec les paramètres donnés"""
        exp_alpha = np.exp(-alpha * delta_t)
        coeff_R = (1 - exp_alpha) / alpha
        
        N = len(R)
        V = np.zeros(N)
        V[0] = Vmax / 2  # Condition initiale
        
        for n in range(N-1):
            V_pred = exp_alpha * V[n] + coeff_R * R[n]
            V[n+1] = min(max(V_pred, 1e-7), Vmax)
        
        return alpha * V  # Q_sim

    def _optimize_criterion(self, cost_function, x0, R, Q, delta_t):
        """Optimisation générique pour différents critères"""
        N = len(Q)
        res = minimize(
            cost_function,
            x0,
            args=(R, Q, delta_t, N),
            method='Nelder-Mead',
            options={
                'xatol': 1e-6,
                'fatol': 1e-6,
                'disp': False
            }
        )
        
        alpha_opt = np.exp(res.x[0])
        Vmax_opt = np.exp(res.x[1])
        criterion_value = -res.fun  # Conversion du coût en valeur positive
        
        return alpha_opt, Vmax_opt, criterion_value

    def crit_NSE_opti(self, R: np.ndarray, Q: np.ndarray, delta_t: float) -> tuple[float, float, float]:
        """Optimise pour maximiser le critère NSE standard"""
        def cost_function(x, R, Q, delta_t, N):
            alpha = np.exp(x[0])
            Vmax = np.exp(x[1])
            Q_sim = self._simulate_reservoir(alpha, Vmax, R, delta_t)
            
            numerator = np.sum((Q - Q_sim)**2)
            denominator = np.sum((Q - np.mean(Q))**2)
            
            if denominator < 1e-12:
                NSE_val = -np.inf
            else:
                NSE_val = 1 - numerator / denominator
            
            return -NSE_val
        
        x0 = np.array([np.log(0.1), np.log(1000)])
        return self._optimize_criterion(cost_function, x0, R, Q, delta_t)

    def crit_NSE_log_opti(self, R: np.ndarray, Q: np.ndarray, delta_t: float) -> tuple[float, float, float]:
        """Optimise pour maximiser le critère NSE sur les logarithmes"""
        def cost_function(x, R, Q, delta_t, N):
            alpha = np.exp(x[0])
            Vmax = np.exp(x[1])
            Q_sim = self._simulate_reservoir(alpha, Vmax, R, delta_t)
            
            # Filtrage des valeurs non-positives
            mask = (Q > 0) & (Q_sim > 0)
            if np.sum(mask) < 10:  # Au moins 10 points
                return np.inf
                
            logQ = np.log(Q[mask])
            logQ_sim = np.log(Q_sim[mask])
            
            numerator = np.sum((logQ - logQ_sim)**2)
            denominator = np.sum((logQ - np.mean(logQ))**2)
            
            if denominator < 1e-12:
                NSE_log_val = -np.inf
            else:
                NSE_log_val = 1 - numerator / denominator
            
            return -NSE_log_val
        
        x0 = np.array([np.log(0.1), np.log(1000)])
        return self._optimize_criterion(cost_function, x0, R, Q, delta_t)

    def crit_KGE_opti(self, R: np.ndarray, Q: np.ndarray, delta_t: float) -> tuple[float, float, float]:
        """Optimise pour maximiser le critère KGE (Kling-Gupta Efficiency)"""
        def cost_function(x, R, Q, delta_t, N):
            alpha = np.exp(x[0])
            Vmax = np.exp(x[1])
            Q_sim = self._simulate_reservoir(alpha, Vmax, R, delta_t)
            
            # Calcul des composantes KGE
            r = np.corrcoef(Q, Q_sim)[0, 1]
            beta = np.mean(Q_sim) / np.mean(Q)
            gamma = (np.std(Q_sim) / np.mean(Q_sim)) / (np.std(Q) / np.mean(Q))
            
            # Formule KGE standard
            KGE = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
            return -KGE  # Minimise le négatif du KGE
        
        x0 = np.array([np.log(0.1), np.log(1000)])
        return self._optimize_criterion(cost_function, x0, R, Q, delta_t)
    




    def crit_RMSE_opti(self) -> float:
        """
        Calcule le Root Mean Squared Error (RMSE) entre Q_obs et Q_sim.

        Returns
        -------
        Valeur du RMSE.
        """
        
        return np.sqrt(np.mean((self.Q_obs - self.Q_sim) ** 2))
    
    def crit_Biais_opti(self) -> float:

        """
        Calcule le biais en pourcentage

        Parametre de sortie :
        Valeur du biais en % 
        """
        somme_obs = np.sum(self.Q_obs)
        if somme_obs == 0:
            return np.nan  # évite division par zéro
        return 100 * np.sum(self.Q_sim - self.Q_obs) / somme_obs




   
    
    def crit_mix_opti(self,  weights: Dict[str, float], transfo: Dict[str, str]) -> float:
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