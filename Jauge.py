  
import os
import pandas as pd
import numpy as np

from Pre_process import Pre_Process

class Jauge :
    """ 
    Donnees pour un bassin versant jauge
    Fonctionne pour l'instant en recuperant les donnees en format csv

    Attributs
    id, nom (str) : identifiant et nom du bassin versant
    csv_dir (str) : repertoire contenant le fichier de donnees
    csv_name (str) : nom du fichier de donnees
    donnees (pd.DataFrame) : fichier de donnes (provenant de la base CAMELS)
    watershed (Pre_process) : objet watershed contenant des informations de base sur le bassin versant
    """

    def __init__(self, id: str, nom: str, csv_dir: str, csv_name: str, watershed:Pre_Process) :

        csv_path = os.path.join(csv_dir, csv_name)

        self.watershed_id = id
        self.nom = nom
        self.donnees = pd.read_csv(csv_path, sep=';', header=7)
        self.watershed = watershed

    def serie_debit(self, start:str, end:str) -> np.ndarray :
        """
        Renvoie la série de débits mesurés entre start et end pour le bassin versant self

        Paramètre d’entrée :
        start : date de début de la période souhaitée (ex. '2005-01-01').
        end : date de fin de la période souhaitée (ex. '2010-12-31').

        Paramètres de sortie :
        un np.ndarray correspondant aux mesures de débits sur cette la période choisie
        """

        self.donnees["DatesR"] = pd.to_datetime(self.donnees["tsd_date"].astype(str), format="%Y%m%d")

        l = (self.donnees["DatesR"].dt.date >= pd.to_datetime(start).date()) & \
            (self.donnees["DatesR"].dt.date <= pd.to_datetime(end).date())
        lignes =  [i for i in self.donnees.index[l]]
        colonnes = ["tsd_q_mm"]
        extrait = self.donnees.loc[lignes, colonnes]
        Q = extrait.iloc[:, 0].to_numpy()

        mask = np.isnan(Q)
        return Q[~mask]
    
    def serie_debit_mensuel(self, start: str, end: str) ->np.ndarray:
        """
        Renvoie la série des débits mensuels cumulés entre start et end
        pour le bassin versant self

        Paramètres d’entrée
        start : Date de début de la période souhaitée (ex. '2005-01-01')
        end : Date de fin de la période souhaitée (ex. '2010-12-31')

        Paramètre de sortie
            array par la fin de chaque mois
            contenant la somme des débits journaliers du mois
        """
        
        df = self.donnees.copy()
        df["DatesR"] = pd.to_datetime(df["tsd_date"].astype(str), format="%Y%m%d")

        mask = (
            (df["DatesR"] >= pd.to_datetime(start)) &
            (df["DatesR"] <= pd.to_datetime(end))
        )
        df = df.loc[mask, :]

        df = df.set_index("DatesR")
        df["tsd_q_mm"] = pd.to_numeric(df["tsd_q_mm"], errors="coerce")

        monthly_sum = df["tsd_q_mm"].resample("ME").sum()

        monthly_sum = monthly_sum.replace(0, np.nan).dropna()

        return monthly_sum.to_numpy()