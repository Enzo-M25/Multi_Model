  
import os
import pandas as pd
import numpy as np

# TODO API
class Jauge :
    """
    Donnees pour un bassin versant jauge
    Fonctionne pour l'instant en recuperant les donnees en format csv

    Attributs
    id : identifiant du bassin versant
    csv_dir : repertoire contenant le fichier de donnees
    csv_name : nom du fichier de donnees
    donnees : fichier de donnes (provenant de la base CAMELS)
    """

    def __init__(self, id: str, nom: str, csv_dir: str, csv_name: str) :

        self.watershed_id = id
        self.nom = nom
        csv_path = os.path.join(csv_dir, csv_name)
        self.donnees = pd.read_csv(csv_path, sep=';', header=7)

    def serie_debit(self, start:str, end:str) -> pd.Series :
        """
        Renvoie la série de débits mesurés entre start et end pour le bassin versant self

        Paramètre d’entrée :
        start : date de début de la période souhaitée (ex. '2005-01-01').
        end : date de fin de la période souhaitée (ex. '2010-12-31').

        Paramètres de sortie :
        un panda.Series correspondant aux mesures de débits sur cette la période choisie
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