  
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional

class Outputs :
    """
    Gere les affichages et donnees en post-processing

    Attributs :
    id : identifiant du bassin versant
    name : nom du bassin versant
    Q_pred : débits simulés pendant la période date sous forme de panda Series
    crit_pred : valeur du critere obtenue lors de la prediction
    Q_obs : débits observés pendant la période date sous forme de panda Series (optionnel si la prediction est realisee a une date future)
    dates : periode de prediction / estimation des debits
    """

    def __init__(self, id: str, name: str, date:pd.Series, Q_pred:pd.Series, Q_obs:Optional[pd.Series] = None, crit: Optional[float] = None) :

        self.watershed_id = id
        self.name = name
        self.Q_pred = Q_pred
        self._Q_obs: Optional[pd.Series] = Q_obs
        self._crit: Optional[float] = crit

        self.dates = self._convert_dates(date)

    @property
    def Q_obs(self) -> pd.Series:
        """
        Accès sécurisé à Q_obs. Lève une erreur si non initialisé

        Paramètre de sortie :
        Q_obs : débits observés pendant la période date sous forme de panda Series
        """

        if self._Q_obs is None:
            raise AttributeError("Q_obs n'a pas été initialisé. Utilisez set_Q_obs() d'abord.")
        return self._Q_obs

    @property
    def crit(self) -> float:
        """
        Accès sécurisé à crit.  Lève une erreur si crit n’a pas été défini.
        """
        if self._crit is None:
            raise AttributeError(
                "crit n'a pas été initialisé. "
            )
        return self._crit

    def has_Q_obs(self) -> bool:
        """
        Vérifie si Q_obs est disponible
        """
        return self._Q_obs is not None
    
    def has_crit(self) -> bool:
        """
        Vérifie si crit est disponible
        """
        return self._crit is not None

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
            # pandas comprend nativement l’ISO 8601, y compris les nanosecondes
            return pd.to_datetime(
                date_series,
                errors='coerce'    # remplace les valeurs invalides par NaT
            )
        except Exception as e:
            raise ValueError(f"Erreur de conversion des dates : {e}")
    
    def affiche(self) -> None:
        """
        Affiche un graphique permettant de comparer les debits observes et estimes sur la periode definie dans dates ainsi que d'afficher la valeur du critere pour l'estimation
        """

        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        if self.has_Q_obs():
            ax.plot(self.dates, self.Q_obs, 'k-', linewidth=1.5, label='Q mesuré (mm/j)')
        ax.plot(self.dates, self.Q_pred, 'r-', linewidth=1.5, label='Q simulé (mm/j)')

        locator = mdates.AutoDateLocator(minticks=6, maxticks=15)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        
        if self.has_crit():
            ax.set_title(f"Débits - {self.name} {self.watershed_id} | crit = {self.crit}", fontsize=14)
        else:
            ax.set_title(f"Débits - {self.name} {self.watershed_id}", fontsize=14)
        ax.set_xlabel('Temps')
        ax.set_ylabel('Débit (mm/j)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        print("affichage \n")
        
        plt.tight_layout()
        plt.show()

    def affiche_avec_filename(self, filename) -> None:
        """
        
        """

        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        if self.has_Q_obs():
            ax.plot(self.dates, self.Q_obs, 'k-', linewidth=1.5, label='Q mesuré (mm/j)')
        ax.plot(self.dates, self.Q_pred, 'r-', linewidth=1.5, label='Q simulé (mm/j)')

        locator = mdates.AutoDateLocator(minticks=6, maxticks=15)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        
        if self.has_crit():
            ax.set_title(f"Débits - {self.name} {self.watershed_id} | crit = {self.crit}", fontsize=14)
        else:
            ax.set_title(f"Débits - {self.name} {self.watershed_id}", fontsize=14)
        ax.set_xlabel('Temps')
        ax.set_ylabel('Débit (mm/j)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        print("affichage \n")
        
        plt.tight_layout()
        
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        

    def affiche_nuage(self) -> None:
        """
        
        """

        fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},figsize=(10,3))
        ax = a0
        ax.plot(self.dates, self.Q_obs, color='k', lw=1, ls='-', zorder=0, label='observed')
        ax.plot(self.dates, self.Q_pred, color='red', lw=1, label='modeled')

        # ax.plot(Rmod.index, Rmod*1000, color='blue', lw=2.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Q / A [mm/month]')
        ax.set_yscale('log')


        ax = a1
        ax.scatter(self.Q_obs, self.Q_pred,
                    s=25, edgecolor='none', alpha=0.75, facecolor='forestgreen')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot((0.0001,100),(0.0001,100), color='grey', zorder=-1)
        ax.set_xlim(0.0001,100)
        ax.set_ylim(0.0001,100)  
        ax.set_xlabel('$Q_{obs}$ / A [mm/month]', fontsize=12)
        ax.set_ylabel('$Q_{sim}$ / A [mm/month]', fontsize=12)
        fig.tight_layout()

        plt.show()