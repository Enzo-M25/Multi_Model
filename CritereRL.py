
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple
from typing import Dict

class CritereRL :
    """
    Regroupe differentes fonctions permettant de calculer des criteres de performances pour le modele de reservoir lineaire

    Attributs
    Q_obs : Vecteur des débits mesurés sur une certaine période
    Q_sim : Vecteur des débits simulés sur une certaine période
    """

    def __init__(self, Q:np.ndarray, R:np.ndarray, delta_t:float):
        
        if Q.shape != R.shape:
            raise ValueError("Q_obs et Q_sim doivent avoir la même longueur")

        self.Q = Q
        self.R = R
        self.delta_t = delta_t

    def crit_NSE(self, obs, sim) -> float :
        """
        Calcule le critere NSE correspondant

        Paramètre de sortie :
        Valeur du NSE
        """

        Q_bar = np.mean(obs)
        num = np.sum((obs - sim) ** 2)
        denom = np.sum((obs - Q_bar) ** 2)
        return 1 - num / denom
    
    def crit_NSE_log(self, obs, sim) -> float :
        """
        Calcule le critère NSE-log
        
        Parametre de sortie :
        Valeur du NSE-log.
        """

        Q_bar = np.mean(obs)
        eps = Q_bar/100
        obs = obs + eps
        sim = sim + eps
        log_obs = np.log(obs)
        log_sim = np.log(sim)

        num = np.sum((log_obs - log_sim) ** 2)
        den = np.sum((log_obs - np.mean(log_obs)) ** 2)
        return 1 - num / den
    
    def crit_RMSE(self, obs, sim) -> float:
        """
        Calcule le Root Mean Squared Error (RMSE) entre Q_obs et Q_sim.

        Returns
        -------
        Valeur du RMSE.
        """
        
        return np.sqrt(np.mean((obs - sim) ** 2))
    
    def crit_KGE(self, obs, sim) -> float:

        """
        Calcule l'indice Kling-Gupta Efficiency (KGE)

        Parametre de sortie
        Valeur du KGE.
        """
        
        mu_obs = obs.mean()
        mu_sim = sim.mean()

        sigma_obs = obs.std(ddof=0)
        sigma_sim = sim.std(ddof=0)

        r = np.corrcoef(obs, sim)[0, 1]

        alpha = sigma_sim / sigma_obs if sigma_obs != 0 else np.nan
        beta  = mu_sim / mu_obs       if mu_obs    != 0 else np.nan

        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    def crit_Biais(self, obs, sim) -> float:

        """
        Calcule le biais en pourcentage

        Parametre de sortie :
        Valeur du biais en % 
        """
        somme_obs = np.sum(obs)
        if somme_obs == 0:
            return np.nan  # évite division par zéro
        return 100 * np.sum(sim - obs) / somme_obs
    
    def _simulate_reservoir(self, alpha: float, Vmax: float) -> np.ndarray:
        """
        Simule le modèle du réservoir avec les paramètres donnés
        """

        exp_alpha = np.exp(-alpha * self.delta_t)
        coeff_R = (1 - exp_alpha) / alpha
        
        N = len(self.R)
        V = np.zeros(N)
        V[0] = Vmax / 2  # Condition initiale
        
        for n in range(N-1):
            V_pred = exp_alpha * V[n] + coeff_R * self.R[n]
            V[n+1] = min(max(V_pred, 1e-7), Vmax)
        
        return alpha * V  # Q_sim
    
    def _evalution_criteria(self, calib) -> Tuple[Callable, int] :
        """
        
        """

        if calib ==  "crit_NSE" :
            evaluation = self.crit_NSE
            type_err = 1
        elif calib == "crit_NSE_log" :
            evaluation = self.crit_NSE_log
            type_err = 1
        elif calib == "crit_RMSE" :
            evaluation = self.crit_RMSE
            type_err = 0
        elif calib == "crit_KGE" :
            evaluation = self.crit_KGE
            type_err = 1
        elif calib == "crit_Biais" :
            evaluation = self.crit_Biais
            type_err = 0
        else :
            raise ValueError("Critère choisi non reconnu")
        
        return evaluation, type_err
    
    def calculate_criteria(self, alpha, Vmax, fct_calib, dict_crit, transfo) -> float :
        """
        
        """

        Q_obs = self.Q

        Q_sim = self._simulate_reservoir(alpha, Vmax)
        crit = 0

        if fct_calib == "crit_mix" :

            items = list(dict_crit.items())
            total_weight = sum(weight for _, weight in items)

            if total_weight == 0:
                raise ValueError("La somme des poids est nulle, impossible de normaliser")

            for (crit_fct, weight), elem in zip(items, transfo) :
                evaluation, type_err = self._evalution_criteria(crit_fct)
                
                obs = Q_obs.astype(float)
                sim = Q_sim.astype(float)

                if elem == "log":
                    Q_bar = np.mean(obs)
                    eps = Q_bar / 100
                    obs = np.log(obs + eps)
                    sim = np.log(sim + eps)

                elif elem == "inv" :
                    obs = 1.0/obs
                    sim = 1.0/sim

                crit += weight * evaluation(obs, sim)
            crit = crit / total_weight


        else :
            evaluation, type_err = self._evalution_criteria(fct_calib)
            elem = transfo[0]

            obs = Q_obs.astype(float)
            sim = Q_sim.astype(float)

            if elem == "log":
                Q_bar = np.mean(obs)
                eps = Q_bar / 100
                obs = np.log(obs + eps)
                sim = np.log(sim + eps)

            elif elem == "inv" :
                obs = 1.0/obs
                sim = 1.0/sim

            crit = evaluation(obs, sim)

        return crit, type_err

    def _erreur_modele_norm(self, x, fct_calib, dict_crit, transfo) -> float :

        alpha, Vmax = x

        crit, type_err = self.calculate_criteria(alpha, Vmax, fct_calib, dict_crit, transfo)

        error = (1 - crit) if type_err == 1 else crit

        if np.isnan(error) :
            return np.inf
        
        return error
    
    def optimize_criterion(self, alpha, Vmax, fct_calib, dict_crit, transfo) -> Tuple[float,float]:
        """
        
        """

        x0 = [alpha, Vmax]
        
        res = minimize(
            self._erreur_modele_norm,
            x0,
            args=(fct_calib, dict_crit, transfo),
            method='Nelder-Mead',
            options={
                'xatol': 1e-6,
                'fatol': 1e-6,
                'disp': False
            }
        )
        
        alpha_opt = res.x[0]
        Vmax_opt = res.x[1]
        
        return alpha_opt, Vmax_opt