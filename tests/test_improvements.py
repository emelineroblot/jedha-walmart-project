# -*- coding: utf-8 -*-
"""Tests du module src/improvements.py (pistes hors perimetre du brief)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src import improvements as imp
from src import walmart as w


@pytest.fixture(scope="module")
def xy():
    df = w.clean(w.load_raw("Walmart_Store_sales.csv"))
    return df.drop(w.TARGET, axis=1), df[w.TARGET]


# --------------------------------------------------------------------------- #
# A2 : encodage par cible
# --------------------------------------------------------------------------- #

def test_target_encoder_remplace_store_par_une_colonne_numerique():
    X = pd.DataFrame({"Store": ["1", "1", "2", "2"], "autre": [1, 2, 3, 4]})
    y = pd.Series([100.0, 200.0, 900.0, 1100.0])
    out = imp.StoreTargetEncoder(smoothing=0.0).fit(X, y).transform(X)
    assert "Store" not in out.columns
    assert imp.STORE_TE_COL in out.columns
    assert out[imp.STORE_TE_COL].tolist() == [150.0, 150.0, 1000.0, 1000.0]


def test_target_encoder_lisse_vers_la_moyenne_generale():
    """Un magasin rare ne doit pas imposer sa moyenne brute."""
    X = pd.DataFrame({"Store": ["1", "1", "1", "1", "2"]})
    y = pd.Series([100.0, 100.0, 100.0, 100.0, 1000.0])
    enc = imp.StoreTargetEncoder(smoothing=10.0).fit(X, y)
    prior = y.mean()
    # le magasin 2 n'a qu'une observation : son encodage doit rester proche du prior
    assert abs(enc.mapping_["2"] - prior) < abs(1000.0 - prior)


def test_target_encoder_magasin_inconnu_recoit_la_moyenne_generale():
    X = pd.DataFrame({"Store": ["1", "2"]})
    y = pd.Series([100.0, 300.0])
    enc = imp.StoreTargetEncoder(smoothing=0.0).fit(X, y)
    out = enc.transform(pd.DataFrame({"Store": ["99"]}))
    assert out[imp.STORE_TE_COL].iloc[0] == pytest.approx(y.mean())
    assert out[imp.STORE_TE_COL].isna().sum() == 0


def test_target_encoder_reduit_le_nombre_de_features(xy):
    X, y = xy
    n_dummies = w.build_pipeline(Ridge())[:-1].fit(X, y).transform(X).shape[1]
    n_te = imp.build_pipeline_target_encoded(Ridge())[:-1].fit(X, y).transform(X).shape[1]
    assert n_te < n_dummies


def test_pipeline_target_encoded_predit_sur_magasin_inconnu(xy):
    X, y = xy
    pipe = imp.build_pipeline_target_encoded(Ridge(alpha=1.0)).fit(X, y)
    inconnu = X.iloc[[0]].copy()
    inconnu["Store"] = "MAGASIN_INCONNU"
    assert np.isfinite(pipe.predict(inconnu)[0])


def test_imputation_a_lieu_avant_l_encodage(xy):
    """L'imputation groupe par Store, qui disparait a l'encodage : l'ordre est critique."""
    X, y = xy
    steps = [name for name, _ in imp.build_pipeline_target_encoded(Ridge()).steps]
    assert steps.index("impute") < steps.index("te")


# --------------------------------------------------------------------------- #
# A1 : transformation log de la target
# --------------------------------------------------------------------------- #

def test_log_target_predit_en_dollars(xy):
    X, y = xy
    pipe = w.build_pipeline(imp.log_target(Ridge(alpha=1.0))).fit(X, y)
    pred = pipe.predict(X)
    # les predictions doivent rester a l'echelle des ventes, pas a celle du log
    assert pred.min() > 1e4
    assert pred.max() < 1e7


def test_log_target_est_bien_inversible():
    y = np.array([2.5e5, 1e6, 2.8e6])
    assert np.allclose(np.expm1(np.log1p(y)), y)
