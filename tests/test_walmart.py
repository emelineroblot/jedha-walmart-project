# -*- coding: utf-8 -*-
"""Tests du module src/walmart.py.

Lancer depuis la racine du depot :  python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from src import walmart as w


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def raw():
    return w.load_raw("Walmart_Store_sales.csv")


@pytest.fixture(scope="module")
def clean_df(raw):
    return w.clean(raw)


@pytest.fixture(scope="module")
def xy(clean_df):
    return clean_df.drop(w.TARGET, axis=1), clean_df[w.TARGET]


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #

def test_store_charge_en_chaine(raw):
    """Store est un identifiant : le laisser en float le ferait entrer dans les correlations."""
    assert raw["Store"].dtype == object
    assert raw["Store"].iloc[0] == str(int(float(raw["Store"].iloc[0])))


def test_store_exclu_des_colonnes_numeriques(raw):
    assert "Store" not in raw.select_dtypes(include=[np.number]).columns


# --------------------------------------------------------------------------- #
# Holiday_Flag : reconstruction deterministe
# --------------------------------------------------------------------------- #

def test_holiday_flag_reproduit_toutes_les_valeurs_connues(raw):
    """Le coeur de la correction : on ne s'appuie sur la reconstruction que si elle est exacte."""
    dates = pd.to_datetime(raw["Date"], format=w.DATE_FORMAT, errors="coerce")
    agree, known = w.validate_holiday_reconstruction(raw, dates)
    assert known > 0
    assert agree == known, f"{known - agree} desaccord(s) sur {known} valeurs connues"


def test_holiday_flag_marque_bien_une_semaine_feriee():
    dates = pd.Series(pd.to_datetime(["2010-11-26", "2011-03-04"]))
    assert w.reconstruct_holiday_flag(dates).tolist() == [1.0, 0.0]


def test_clean_ne_laisse_aucun_holiday_flag_manquant(clean_df):
    assert clean_df["Holiday_Flag"].isna().sum() == 0


# --------------------------------------------------------------------------- #
# Nettoyage
# --------------------------------------------------------------------------- #

def test_clean_ne_supprime_que_target_et_date(raw, clean_df):
    attendu = int((raw[w.TARGET].notna() & raw["Date"].notna()).sum())
    assert len(clean_df) == attendu


def test_clean_conserve_les_lignes_a_valeurs_manquantes(clean_df):
    """Elles doivent survivre au nettoyage : c'est le pipeline qui les impute."""
    assert clean_df[w.IMPUTE_COLS].isna().any(axis=1).sum() > 0


def test_clean_conserve_plus_que_dropna_global(raw, clean_df):
    naif = len(raw.dropna())
    assert len(clean_df) > naif


def test_pas_de_colonne_constante_dans_les_features(clean_df):
    """DayOfWeek etait constante : aucune feature ne doit l'etre."""
    features = [c for c in clean_df.columns if c != w.TARGET]
    constantes = [c for c in features if clean_df[c].nunique(dropna=True) <= 1]
    assert constantes == []


def test_day_et_dayofweek_absentes(clean_df):
    assert "Day" not in clean_df.columns
    assert "DayOfWeek" not in clean_df.columns


def test_encodage_cyclique_borne_et_continu(clean_df):
    for col in ["Month_sin", "Month_cos", "Week_sin", "Week_cos"]:
        assert clean_df[col].between(-1, 1).all()
    # decembre et janvier doivent etre proches, contrairement a un encodage 12 vs 1
    dec = np.array([np.sin(2 * np.pi * 12 / 12), np.cos(2 * np.pi * 12 / 12)])
    jan = np.array([np.sin(2 * np.pi * 1 / 12), np.cos(2 * np.pi * 1 / 12)])
    jui = np.array([np.sin(2 * np.pi * 6 / 12), np.cos(2 * np.pi * 6 / 12)])
    assert np.linalg.norm(dec - jan) < np.linalg.norm(dec - jui)


# --------------------------------------------------------------------------- #
# Regle des 3-sigma
# --------------------------------------------------------------------------- #

def test_3sigma_bornes_symetriques():
    X = pd.DataFrame({"CPI": [10.0, 20.0, 30.0]})
    low, high = w.fit_3sigma_bounds(X, cols=["CPI"])["CPI"]
    assert low == pytest.approx(20 - 3 * X["CPI"].std())
    assert high == pytest.approx(20 + 3 * X["CPI"].std())


def test_3sigma_conserve_les_nan():
    """Piege : between() renvoie False sur un NaN, ce qui supprimerait les lignes a imputer."""
    X = pd.DataFrame({"CPI": [10.0, np.nan, 1000.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    Xf, yf, n = w.apply_3sigma(X, y, {"CPI": (0.0, 100.0)})
    assert n == 1
    assert Xf["CPI"].isna().sum() == 1
    assert len(Xf) == 2 and len(yf) == 2


def test_3sigma_aligne_X_et_y():
    X = pd.DataFrame({"CPI": [5.0, 500.0, 7.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    Xf, yf, _ = w.apply_3sigma(X, y, {"CPI": (0.0, 100.0)})
    assert list(Xf.index) == list(yf.index)
    assert yf.tolist() == [1.0, 3.0]


def test_3sigma_bornes_estimees_sur_le_train_seul(xy):
    """Deux sous-ensembles differents doivent donner des bornes differentes."""
    X, _ = xy
    b1 = w.fit_3sigma_bounds(X.iloc[:40])
    b2 = w.fit_3sigma_bounds(X.iloc[40:])
    assert b1["Unemployment"] != b2["Unemployment"]


# --------------------------------------------------------------------------- #
# Imputation
# --------------------------------------------------------------------------- #

def test_imputer_utilise_la_mediane_du_magasin():
    X = pd.DataFrame({
        "Store": ["1", "1", "1", "2"],
        "CPI": [100.0, 200.0, np.nan, 999.0],
    })
    out = w.StoreMedianImputer(cols=["CPI"]).fit(X).transform(X)
    assert out["CPI"].iloc[2] == 150.0  # mediane du magasin 1, pas la mediane globale


def test_imputer_repli_sur_mediane_globale_si_magasin_inconnu():
    train = pd.DataFrame({"Store": ["1", "1"], "CPI": [100.0, 200.0]})
    test = pd.DataFrame({"Store": ["99"], "CPI": [np.nan]})
    out = w.StoreMedianImputer(cols=["CPI"]).fit(train).transform(test)
    assert out["CPI"].iloc[0] == 150.0
    assert out["CPI"].isna().sum() == 0


def test_imputer_ne_modifie_pas_l_entree():
    X = pd.DataFrame({"Store": ["1", "1"], "CPI": [100.0, np.nan]})
    w.StoreMedianImputer(cols=["CPI"]).fit(X).transform(X)
    assert X["CPI"].isna().sum() == 1


def test_pipeline_ne_laisse_aucun_nan(xy):
    X, y = xy
    pipe = w.build_pipeline(LinearRegression()).fit(X, y)
    assert not np.isnan(pipe[:-1].transform(X)).any()


# --------------------------------------------------------------------------- #
# Encodage des categories
# --------------------------------------------------------------------------- #

def test_categorie_inconnue_distincte_de_la_reference():
    """Sans drop='first', un magasin inconnu ne doit pas etre encode comme la reference."""
    train = pd.DataFrame({"Store": ["1", "2", "3"], "Holiday_Flag": [0.0, 0.0, 1.0]})
    enc = w.build_preprocessor(numerical=[], categorical=["Store", "Holiday_Flag"]).fit(train)
    ref = pd.DataFrame({"Store": ["1"], "Holiday_Flag": [0.0]})
    inconnu = pd.DataFrame({"Store": ["99"], "Holiday_Flag": [0.0]})
    assert not np.array_equal(enc.transform(ref), enc.transform(inconnu))


def test_categorie_inconnue_ne_fait_pas_planter(xy):
    X, y = xy
    pipe = w.build_pipeline(Ridge(alpha=0.01)).fit(X, y)
    inconnu = X.iloc[[0]].copy()
    inconnu["Store"] = "MAGASIN_INCONNU"
    assert np.isfinite(pipe.predict(inconnu)[0])


# --------------------------------------------------------------------------- #
# Metriques
# --------------------------------------------------------------------------- #

def test_r2_ajuste_penalise_les_parametres():
    assert w.adjusted_r2(0.9, n=100, p=1) > w.adjusted_r2(0.9, n=100, p=50)


def test_r2_ajuste_indefini_si_trop_de_parametres():
    assert np.isnan(w.adjusted_r2(0.9, n=10, p=10))


def test_bootstrap_encadre_le_point():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.3, size=200)
    point, low, high, dist = w.bootstrap_ci(y_true, y_pred, n_boot=300)
    assert low < point < high
    assert len(dist) > 0


def test_bootstrap_reproductible():
    y_true = np.arange(50.0)
    y_pred = y_true + 1
    a = w.bootstrap_ci(y_true, y_pred, n_boot=100)[:3]
    b = w.bootstrap_ci(y_true, y_pred, n_boot=100)[:3]
    assert a == b


# --------------------------------------------------------------------------- #
# Protocole d'evaluation
# --------------------------------------------------------------------------- #

def test_cv_evaluate_renvoie_un_score_par_fold(xy):
    X, y = xy
    from sklearn.model_selection import KFold
    res = w.cv_evaluate(lambda: Ridge(alpha=0.01), X, y, cv=KFold(3, shuffle=True, random_state=0))
    assert res["n_folds"] == 3
    assert len(res["r2_folds"]) == 3
    assert res["r2_min"] <= res["r2_mean"] <= res["r2_max"]


def test_la_regularisation_ameliore_la_generalisation(xy):
    """Avec 29 features pour ~90 lignes, le modele non regularise doit generaliser moins bien."""
    X, y = xy
    from sklearn.model_selection import KFold
    cv = KFold(5, shuffle=True, random_state=42)
    lin = w.cv_evaluate(LinearRegression, X, y, cv=cv)
    ridge = w.cv_evaluate(lambda: Ridge(alpha=0.01), X, y, cv=cv)
    assert ridge["r2_mean"] > lin["r2_mean"]
