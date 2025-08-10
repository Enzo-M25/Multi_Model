  
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional

class Outputs :
    """
    Gere les affichages et donnees en post-processing

    Attributs :
    id (str) : identifiant du bassin versant
    name (str) : nom du bassin versant
    Q_pred (np.ndarray) : débits simulés pendant la période date
    Q_obs (np.ndarray) : débits observés pendant la période date (optionnel si la prediction est realisee a une date future)
    dates (pd.DatetimeIndex) : periode de prediction / estimation des debits
    """

    def __init__(self, id: str, name: str, figures_dir:str, date:pd.Series, Q_pred:np.ndarray, Q_obs:Optional[np.ndarray] = None) :

        self.watershed_id = id
        self.name = name
        self.figures_dir = figures_dir
        self.Q_pred = Q_pred
        self._Q_obs: Optional[np.ndarray] = Q_obs

        self.dates = self._convert_dates(date)

    @property
    def Q_obs(self) -> np.ndarray:
        """
        Accès sécurisé à Q_obs. Lève une erreur si non initialisé

        Paramètre de sortie :
        Q_obs : débits observés pendant la période date sous forme de np.ndarray
        """

        if self._Q_obs is None:
            raise AttributeError("Q_obs n'a pas été initialisé. Utilisez set_Q_obs() d'abord.")
        return self._Q_obs

    def has_Q_obs(self) -> bool:
        """
        Vérifie si Q_obs est disponible
        """
        return self._Q_obs is not None

    def _convert_dates(self, date_series: pd.Series) -> pd.DatetimeIndex:
        """
        Transforme un vecteur de dates en un format datetime lisible.

        Paramètre d'entrée :
        date_series : vecteur de dates sous format ISO 8601
                    ex. '2005-01-01T00:00:00.000000000'

        Paramètre de sortie :
        DatetimeIndex pandas correspondant
        """
        try:
            if isinstance(date_series.index, pd.DatetimeIndex):
                return date_series.index
            return pd.DatetimeIndex(pd.to_datetime(date_series))
        except Exception as e:
            raise ValueError(f"Erreur de conversion des dates : {e}")
        
    def detect_frequency(self, idx: pd.DatetimeIndex) -> str:
        """
        Détecte la fréquence à partir d’un DatetimeIndex.
        Retourne 'daily' si c’est quotidien,
        'monthly' si c'est mensuel,
        sinon 'unknown'.
        """
        
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError(f"L'index doit être un DatetimeIndex, pas {type(idx)}")
        
        diff_days = idx.to_series().diff().dt.days.median()
        
        # Détermination de la fréquence
        if diff_days == 1:
            return 'daily'
        elif 28 <= diff_days <= 31:
            return 'monthly'
        else:
            return 'unknown'
    
    def affiche(self) -> None:
        """
        Affiche un graphique permettant de comparer les debits observes et estimes sur la periode definie dans dates
        """

        freq = self.detect_frequency(self.dates)
        if freq == "daily" :
            label = "(mm/j)"
        elif freq == "monthly" :
            label = "(mm/month)"
        else :
            label = ""


        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        if self.has_Q_obs():
            ax.plot(self.dates, self.Q_obs, 'k-', linewidth=1.5, label=f"Q mesuré {label}")
        ax.plot(self.dates, self.Q_pred, 'r-', linewidth=1.5, label=f"Q simulé {label}")

        locator = mdates.AutoDateLocator(minticks=6, maxticks=15)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        
        if self.has_Q_obs() :
            ax.set_title(f"Prévision et comparaison des débits mesurés et simulés sur le bassin versant {self.name} - {self.watershed_id}", fontsize=14)
            plot_name = f"Prévisions_{self.name}_Q_obs_Q_sim.png"
        else :
            ax.set_title(f"Prévision des débits sur le bassin versant {self.name} - {self.watershed_id}", fontsize=14)
            plot_name = f"Prévisions_{self.name}.png"
        ax.set_xlabel("Dates")
        ax.set_ylabel(f"Débit {label}")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        print("affichage \n")
        
        plt.tight_layout()
        plt.show()

        fig_path = os.path.join(self.figures_dir, plot_name)
        fig.savefig(fig_path, bbox_inches='tight')

        plt.close(fig)

    def affiche_nuage(self) -> None:
        """
        Affiche un graphique permettant de comparer les debits observes et estimes sur la periode definie dans dates ainsi qu'un graphique "ratio" entre simulé et observé
        """

        if not self.has_Q_obs():
            raise AttributeError("Cette méthode nécessite des débits observés (Q_obs)")

        freq = self.detect_frequency(self.dates)
        if freq == "daily" :
            label = "(mm/j)"
        elif freq == "monthly" :
            label = "(mm/month)"
        else :
            label = ""

        fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},figsize=(10,3))
        ax = a0
        ax.plot(self.dates, self.Q_obs, color='k', lw=1, ls='-', zorder=0, label=f"Q mesuré {label}")
        ax.plot(self.dates, self.Q_pred, color='red', lw=1, label=f"Q simulé {label}")

        # ax.plot(Rmod.index, Rmod*1000, color='blue', lw=2.5)
        ax.set_xlabel('Dates')
        ax.set_ylabel(f'Q / A {label}')
        ax.set_yscale('log')

        ax = a1
        ax.scatter(self.Q_obs, self.Q_pred,
                    s=25, edgecolor='none', alpha=0.75, facecolor='forestgreen')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot((0.0001,100),(0.0001,100), color='grey', zorder=-1)
        ax.set_xlim(0.0001,100)
        ax.set_ylim(0.0001,100)  
        ax.set_xlabel(f"Q_obs / A {label}", fontsize=12)
        ax.set_ylabel(f"Q_sim / A {label}", fontsize=12)
        fig.suptitle(f"Prévision et comparaison des débits mesurés et simulés sur le bassin versant {self.name} - {self.watershed_id}", fontsize=14)
        fig.tight_layout()

        plt.show()

        fig_path = os.path.join(self.figures_dir, f"Prévisions_comparaisons_{self.name}_Q_obs_Q_sim.png")
        fig.savefig(fig_path, bbox_inches='tight')

        plt.close(fig)