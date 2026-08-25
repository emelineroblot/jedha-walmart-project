# -*- coding: utf-8 -*-
"""Fonctions et transformateurs reutilisables du projet Walmart Sales Prediction.

Ce module regroupe tout ce qui est testable independamment du notebook :
chargement, nettoyage, filtrage des outliers, imputation, construction du
pipeline et protocole d'evaluation.

Les tests correspondants sont dans tests/test_walmart.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Constantes du projet
# --------------------------------------------------------------------------- #

#: Semaines feriees du dataset Walmart (Super Bowl, Labor Day, Thanksgiving, Noel).
#: Connaissance metier externe : aucune estimation a partir des donnees.
HOLIDAY_WEEKS = frozenset(
    pd.Timestamp(d)
    for d in [
        "2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08",  # Super Bowl
        "2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06",  # Labor Day
        "2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29",  # Thanksgiving
        "2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27",  # Noel
    ]
)

TARGET = "Weekly_Sales"
DATE_FORMAT = "%d-%m-%Y"

#: Colonnes soumises a la regle des 3-sigma (imposee par le brief).
OUTLIER_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]

#: Colonnes imputees par mediane de magasin.
IMPUTE_COLS = list(OUTLIER_COLS)

CATEGORICAL_FEATURES = ["Store", "Holiday_Flag"]
#: Variables numeriques, exactement celles listees par l'enonce.
NUMERICAL_FEATURES = [
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Year", "Month", "Day", "DayOfWeek",
]


# --------------------------------------------------------------------------- #
# Chargement et nettoyage
# --------------------------------------------------------------------------- #

def load_raw(path: str = "Walmart_Store_sales.csv") -> pd.DataFrame:
    """Charge le CSV brut. `Store` est un identifiant : charge en chaine."""
    df = pd.read_csv(path)
    df["Store"] = df["Store"].astype("Int64").astype(str)
    return df


def reconstruct_holiday_flag(dates: pd.Series) -> pd.Series:
    """Reconstruit Holiday_Flag depuis Date. Fonction deterministe, pas une imputation."""
    return dates.isin(HOLIDAY_WEEKS).astype(float)


def validate_holiday_reconstruction(df: pd.DataFrame, dates: pd.Series) -> tuple[int, int]:
    """Compare la reconstruction aux valeurs connues.

    Returns:
        (nombre d'accords, nombre de valeurs connues)
    """
    known = df["Holiday_Flag"].notna() & dates.notna()
    agree = int((reconstruct_holiday_flag(dates)[known] == df.loc[known, "Holiday_Flag"]).sum())
    return agree, int(known.sum())


def add_time_features(df: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Cree les quatre features temporelles demandees par l'enonce.

    year, month, day et day of week, en numerique.

    Note d'analyse : sur ce dataset, `DayOfWeek` est CONSTANTE — toutes les
    dates sont des vendredis. La colonne est conservee par conformite a
    l'enonce, mais elle ne porte aucune information : le StandardScaler la
    ramene a zero et le Lasso lui attribue un coefficient nul. Le controle est
    fait explicitement dans le notebook plutot que suppose.
    """
    df = df.copy()
    df["Year"] = dates.dt.year
    df["Month"] = dates.dt.month
    df["Day"] = dates.dt.day
    df["DayOfWeek"] = dates.dt.dayofweek
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage complet, hors filtrage des outliers (qui suit le split).

    On ne supprime que ce qui n'est pas recuperable : la target (jamais imputee)
    et la date (source des features temporelles). Le reste est impute dans le
    pipeline, donc appris fold par fold.
    """
    out = df.dropna(subset=[TARGET]).copy()
    dates = pd.to_datetime(out["Date"], format=DATE_FORMAT, errors="raise")
    keep = dates.notna()
    out, dates = out[keep], dates[keep]

    out["Holiday_Flag"] = reconstruct_holiday_flag(dates)
    out = add_time_features(out, dates)
    return out.drop(columns=["Date"])


# --------------------------------------------------------------------------- #
# Regle des 3-sigma
# --------------------------------------------------------------------------- #

def fit_3sigma_bounds(X: pd.DataFrame, cols=OUTLIER_COLS) -> dict:
    """Estime les bornes [mean - 3*std, mean + 3*std].

    A n'appeler que sur des donnees d'entrainement : ces bornes sont un
    parametre appris, au meme titre que la moyenne d'un StandardScaler.
    """
    return {c: (X[c].mean() - 3 * X[c].std(), X[c].mean() + 3 * X[c].std()) for c in cols}


def apply_3sigma(X: pd.DataFrame, y: pd.Series, bounds: dict):
    """Applique des bornes deja estimees.

    Les NaN sont CONSERVES : ils seront imputes dans le pipeline. Sans cela,
    `between()` renvoyant False pour un NaN, le filtre supprimerait exactement
    les lignes que l'imputation doit sauver.

    Returns:
        (X filtre, y filtre, nombre de lignes supprimees)
    """
    mask = np.ones(len(X), dtype=bool)
    for col, (low, high) in bounds.items():
        mask &= (X[col].between(low, high) | X[col].isna()).to_numpy()
    return X[mask], y[mask], int((~mask).sum())


# --------------------------------------------------------------------------- #
# Transformateurs
# --------------------------------------------------------------------------- #

class StoreMedianImputer(BaseEstimator, TransformerMixin):
    """Impute par la mediane du magasin, avec repli sur la mediane globale.

    Les indicateurs economiques sont regionaux : pour un meme magasin a une date
    proche, la valeur est quasi identique. Une mediane globale ecraserait cette
    structure.

    Les medianes sont apprises dans `fit()`, donc re-apprises a chaque fold.
    """

    def __init__(self, cols=None, group="Store"):
        self.cols = cols
        self.group = group

    def fit(self, X, y=None):
        cols = self.cols if self.cols is not None else IMPUTE_COLS
        self.cols_ = list(cols)
        self.by_group_ = {c: X.groupby(self.group)[c].median() for c in self.cols_}
        self.global_ = {c: X[c].median() for c in self.cols_}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X[c] = X[c].fillna(X[self.group].map(self.by_group_[c])).fillna(self.global_[c])
        return X


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def build_preprocessor(numerical=None, categorical=None) -> ColumnTransformer:
    """Normalisation des numeriques, encodage one-hot des categorielles.

    Pas de `drop='first'` : combine a `handle_unknown='ignore'`, il rendrait une
    categorie inconnue indistinguable de la categorie de reference.
    """
    numerical = NUMERICAL_FEATURES if numerical is None else numerical
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical),
        ]
    )


def build_pipeline(model, numerical=None, categorical=None) -> Pipeline:
    """Imputation + preprocessing + modele, en un seul objet reutilisable."""
    return Pipeline(
        [
            ("impute", StoreMedianImputer()),
            ("pre", build_preprocessor(numerical, categorical)),
            ("model", model),
        ]
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def adjusted_r2(r2: float, n: int, p: int) -> float:
    """R2 ajuste : penalise le nombre de parametres, contrairement au R2 brut."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan


def make_cv(n_splits: int = 5, n_repeats: int = 10, random_state: int = 42) -> RepeatedKFold:
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def cv_evaluate(make_model, X, y, cv=None, pipeline_factory=build_pipeline) -> dict:
    """Validation croisee sans aucune fuite de donnees.

    A chaque fold : bornes 3-sigma re-estimees sur le fold d'entrainement,
    filtrage de ce seul fold, puis fit du pipeline complet. Le fold de
    validation n'est ni filtre ni utilise pour apprendre quoi que ce soit.

    C'est la raison pour laquelle la boucle est ecrite a la main plutot qu'avec
    `cross_val_score` : ce dernier ne permet pas de re-estimer les bornes
    d'outliers par fold.
    """
    cv = make_cv() if cv is None else cv
    r2s, rmses, maes, r2_trains, adj_trains = [], [], [], [], []

    for train_idx, val_idx in cv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        X_tr, y_tr, _ = apply_3sigma(X_tr, y_tr, fit_3sigma_bounds(X_tr))

        pipe = pipeline_factory(make_model()).fit(X_tr, y_tr)
        y_val_pred = pipe.predict(X_val)

        r2s.append(r2_score(y_val, y_val_pred))
        rmses.append(root_mean_squared_error(y_val, y_val_pred))
        maes.append(mean_absolute_error(y_val, y_val_pred))

        r2_tr = r2_score(y_tr, pipe.predict(X_tr))
        n_features = pipe[:-1].transform(X_tr).shape[1]
        r2_trains.append(r2_tr)
        adj_trains.append(adjusted_r2(r2_tr, len(X_tr), n_features))

    return {
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "r2_min": float(np.min(r2s)), "r2_max": float(np.max(r2s)),
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "r2_train_mean": float(np.mean(r2_trains)),
        "r2_adj_train": float(np.nanmean(adj_trains)),
        "r2_folds": np.asarray(r2s),
        "n_folds": len(r2s),
    }


def print_cv(name: str, s: dict) -> None:
    print(f"=== {name} ===")
    print(f"  R2   CV : {s['r2_mean']:.4f} +/- {s['r2_std']:.4f}   "
          f"(min {s['r2_min']:.3f} / max {s['r2_max']:.3f} sur {s['n_folds']} folds)")
    print(f"  RMSE CV : {s['rmse_mean']:,.0f} $ +/- {s['rmse_std']:,.0f} $")
    print(f"  MAE  CV : {s['mae_mean']:,.0f} $")
    print(f"  R2 train : {s['r2_train_mean']:.4f}  |  R2 ajuste train : {s['r2_adj_train']:.4f}")
    print(f"  Ecart train/validation : {s['r2_train_mean'] - s['r2_mean']:.4f}")
    print()


def bootstrap_ci(y_true, y_pred, metric=r2_score, n_boot: int = 2000,
                 random_state: int = 42, level: float = 0.95):
    """Intervalle de confiance par bootstrap sur un jeu de donnees.

    Sur un hold-out de petite taille, un score ponctuel est trompeur : le
    reechantillonnage montre l'etendue reellement compatible avec les donnees.

    Returns:
        (point, borne basse, borne haute, distribution)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # un rechantillon degenere (toutes les valeurs identiques) rend R2 indefini
        if np.ptp(y_true[idx]) == 0:
            continue
        stats.append(metric(y_true[idx], y_pred[idx]))

    stats = np.asarray(stats)
    tail = (1 - level) / 2 * 100
    return (
        float(metric(y_true, y_pred)),
        float(np.percentile(stats, tail)),
        float(np.percentile(stats, 100 - tail)),
        stats,
    )
