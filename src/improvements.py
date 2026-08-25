# -*- coding: utf-8 -*-
"""Pistes d'amelioration explorees HORS du perimetre de l'enonce.

Ce module est volontairement separe de `walmart.py`, qui contient le pipeline
du livrable et s'en tient strictement au brief. Rien ici n'est utilise par
01-Walmart_sales.ipynb : tout est exerce depuis 02-Walmart_ameliorations.ipynb.

Pistes couvertes :
    A1  transformation log de la target
    A2  encodage par cible des magasins, en remplacement des dummies
    A3  modeles non lineaires en reference
    A4  ElasticNet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .walmart import NUMERICAL_FEATURES, StoreMedianImputer

STORE_TE_COL = "Store_te"


class StoreTargetEncoder(BaseEstimator, TransformerMixin):
    """Remplace `Store` par la moyenne lissee des ventes du magasin (piste A2).

    Les 20 dummies de magasin consomment l'essentiel du budget de parametres.
    L'encodage par cible les ramene a une seule colonne numerique, et supprime
    du meme coup la notion de « categorie inconnue » : un magasin jamais vu
    recoit la moyenne generale au lieu d'un vecteur nul.

    Le lissage bayesien evite qu'un magasin vu deux fois impose sa moyenne
    brute : plus un magasin est rare, plus son encodage est tire vers la
    moyenne generale.

        encodage = w * moyenne_magasin + (1 - w) * moyenne_generale
        avec w = n_magasin / (n_magasin + smoothing)

    Limite connue : l'encodage d'une ligne utilise la cible de cette ligne. Le
    lissage attenue la fuite sans la supprimer ; un encodage impute par
    sous-validation croisee serait plus rigoureux. Les medianes etant apprises
    dans `fit()`, il n'y a en revanche aucune fuite ENTRE folds.
    """

    def __init__(self, col: str = "Store", smoothing: float = 10.0):
        self.col = col
        self.smoothing = smoothing

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.prior_ = float(y.mean())
        stats = y.groupby(X[self.col]).agg(["mean", "count"])
        weight = stats["count"] / (stats["count"] + self.smoothing)
        self.mapping_ = weight * stats["mean"] + (1 - weight) * self.prior_
        return self

    def transform(self, X):
        X = X.copy()
        X[STORE_TE_COL] = X[self.col].map(self.mapping_).astype(float).fillna(self.prior_)
        return X.drop(columns=[self.col])


def build_pipeline_target_encoded(model, smoothing: float = 10.0) -> Pipeline:
    """Pipeline A2 : encodage par cible des magasins a la place des dummies.

    L'imputation vient EN PREMIER : elle groupe par `Store`, qui n'existe plus
    apres l'encodage.
    """
    numerical = list(NUMERICAL_FEATURES) + [STORE_TE_COL]
    return Pipeline(
        [
            ("impute", StoreMedianImputer()),
            ("te", StoreTargetEncoder(smoothing=smoothing)),
            (
                "pre",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), numerical),
                        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                         ["Holiday_Flag"]),
                    ]
                ),
            ),
            ("model", model),
        ]
    )


def log_target(model) -> TransformedTargetRegressor:
    """Enveloppe un modele pour qu'il apprenne sur log1p(y) (piste A1).

    Les ventes s'etalent de 2.7e5 a 2.8e6 $. Travailler sur log1p stabilise la
    variance et rend l'echelle d'alpha naturelle. Les predictions sont
    re-exprimees en dollars par expm1, donc toutes les metriques restent
    comparables a celles du livrable.
    """
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
