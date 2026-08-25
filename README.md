# 🛒 Walmart Sales Prediction

Projet de **Machine Learning supervisé** : prédire les ventes hebdomadaires des magasins Walmart à partir d'indicateurs économiques.

---

## 📋 Contexte

Le service marketing de Walmart souhaite un modèle capable d'estimer les ventes hebdomadaires en magasin, afin de :

- comprendre comment les ventes sont influencées par les indicateurs économiques ;
- planifier les futures campagnes marketing.

Le projet suit les trois parties demandées : **EDA et preprocessing**, **régression linéaire (baseline)**, **régularisation (Ridge / Lasso)**.

---

## 🎯 Résultat

| | |
|---|---|
| **Modèle retenu** | `Ridge(alpha=0.001)` dans un `Pipeline` complet |
| **Performance** | **R² = 0.865 ± 0.268** (validation croisée, 50 folds) |
| **RMSE** | 208 851 $ |
| **Hold-out** (24 lignes, indicatif) | R² = 0.755, IC 95 % **[0.09 ; 0.98]** · RMSE = 292 296 $ |

> ⚠️ **Le chiffre à retenir est le R² de validation croisée, pas celui du hold-out.**
> Avec un dataset de cette taille, un score mesuré sur un seul découpage n'est pas une mesure de performance. Le bootstrap le rend visible : l'intervalle de confiance à 95 % du R² sur le hold-out s'étend de **0.09 à 0.98**, soit une largeur de 0.89 point. C'est pourquoi tous les scores sont reportés en **moyenne ± écart-type** sur 50 folds.

### Comparaison des modèles

| Modèle | R² CV | Écart-type | Pire fold | RMSE CV |
|---|---|---|---|---|
| **Ridge GridSearch (α=0.001)** | **0.865** | 0.268 | -0.781 | 208 851 $ |
| Lasso GridSearch (α=10) | 0.856 | 0.305 | -1.109 | 213 191 $ |
| Linear Regression | 0.845 | 0.345 | -1.227 | 215 829 $ |
| Ridge (α=1) | 0.841 | **0.079** | **+0.542** | 258 695 $ |
| Lasso (α=1) | 0.837 | 0.361 | -1.391 | 221 639 $ |

`Ridge(α=1)` attire l'œil : son écart-type est trois fois plus faible et il ne descend jamais sous +0.54. Le notebook tranche par un **test apparié** sur les 50 mêmes folds : le modèle retenu gagne **46 fois sur 50** (gain médian +0.059, Wilcoxon p = 4.6e-07), et son **10ᵉ centile est meilleur** (0.882 contre 0.748). Son écart-type plus élevé tient à un petit nombre de folds, pas à une instabilité générale — ici, l'écart-type est un résumé trompeur.

La **régression linéaire non régularisée reste inutilisable** : R² d'entraînement 0.975 contre 0.845 en validation, et un pire fold à **-1.23**, c'est-à-dire moins bon que prédire simplement la moyenne des ventes. La régularisation n'est pas un raffinement ici, elle est nécessaire.

---

## 📊 Dataset

150 lignes, 8 colonnes, avec des valeurs manquantes sur presque toutes les colonnes (8 à 12 %).

| Colonne | Type | Description |
|---|---|---|
| `Store` | Catégorielle | Identifiant du magasin (1-20) — chargé en **chaîne**, c'est un identifiant et non une grandeur |
| `Date` | Date | Date de la semaine (format `dd-mm-yyyy`) |
| `Weekly_Sales` | Numérique | **Target** — ventes hebdomadaires en $ |
| `Holiday_Flag` | Catégorielle | Semaine fériée (0/1) |
| `Temperature` | Numérique | Température moyenne (°F) |
| `Fuel_Price` | Numérique | Prix du carburant ($) |
| `CPI` | Numérique | Indice des prix à la consommation |
| `Unemployment` | Numérique | Taux de chômage (%) |

---

## 🛠️ Méthodologie

### Partie 1 — EDA et preprocessing

#### Exploration
Statistiques descriptives, analyse des valeurs manquantes, et visualisations : distribution de la target, matrice de corrélation (**`Store` exclu** — corréler un numéro de magasin au chiffre d'affaires n'a pas de sens), boxplots, ventes par magasin, impact des jours fériés.

Un contrôle explicite vérifie que le nettoyage ne déforme pas la population étudiée : l'écart médian des ventes moyennes par magasin entre données brutes et nettoyées est de **0.4 %**, et aucun magasin n'est perdu. Les conclusions de l'EDA restent donc valables pour les données effectivement modélisées.

#### Nettoyage — on ne supprime que ce qui n'est pas récupérable

```
150 lignes
 → 136   suppression des NaN sur Weekly_Sales (on n'impute jamais une target)
 → 118   suppression des NaN sur Date (source des features temporelles)
 → split 94 train / 24 hold-out
 →  90   filtre 3-sigma appliqué au train (4 lignes sur Unemployment)
         hold-out laissé intact : 24 lignes
```

Le reste des valeurs manquantes est **imputé dans le `Pipeline`**, donc appris fold par fold :

- **`Holiday_Flag` n'est pas imputée, elle est reconstruite.** C'est une fonction déterministe de `Date` (semaines Super Bowl / Labor Day / Thanksgiving / Noël). La reconstruction est **validée par assertion** avant usage : `109/109` valeurs connues reproduites exactement.
- **`CPI`, `Unemployment`, `Fuel_Price`, `Temperature`** sont des indicateurs régionaux : imputés par la **médiane du magasin** (`StoreMedianImputer`), avec repli sur la médiane globale.

#### Features temporelles

Les quatre features demandées par l'énoncé : `Year`, `Month`, `Day`, `DayOfWeek`.

Un **contrôle de variance** accompagne leur création : `DayOfWeek` est **constante** sur ce dataset — toutes les dates sont des vendredis. La colonne est conservée par conformité à l'énoncé, en notant qu'elle ne peut rien apporter. Le Lasso le confirme plus loin en lui attribuant un coefficient nul.

#### Règle des 3-sigma

Imposée par le brief : sont considérées comme outliers les valeurs hors de $[\bar{X} - 3\sigma, \bar{X} + 3\sigma]$ sur `Temperature`, `Fuel_Price`, `CPI` et `Unemployment`.

Les bornes sont **estimées sur le seul jeu d'entraînement** et ré-estimées à l'intérieur de chaque fold de validation croisée. Le filtre conserve volontairement les `NaN`, qui seront imputés ensuite.

#### Preprocessing

Un `Pipeline` unique, réappris intégralement à chaque fold :

```python
Pipeline([
    ('impute', StoreMedianImputer()),
    ('pre', ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
    ])),
    ('model', model),
])
```

28 features pour 90 lignes d'entraînement (ratio p/n = 0.31).

### Partie 2 — Régression linéaire (baseline)

Entraînement, évaluation par validation croisée (RMSE, MAE, R², **R² ajusté**), diagnostic d'overfitting, interprétation des coefficients, et **graphe des résidus** calculé sur des prédictions hors échantillon.

### Partie 3 — Régularisation

`Ridge` et `Lasso`, puis optimisation d'`alpha` par `GridSearchCV`, comparaison finale avec barres d'erreur, et évaluation unique sur le hold-out.

---

## 🔍 Points de méthode

### Le protocole d'évaluation

- **Comparaison des modèles** : `RepeatedKFold(n_splits=5, n_repeats=10)` = 50 folds, scores en moyenne ± écart-type, avec min et max.
- **Sélection d'`alpha`** : sur le score de validation croisée (`GridSearchCV` reçoit le `Pipeline` complet et les **données brutes**, pas une matrice déjà transformée).
- **Hold-out** : consommé **une seule fois**, en dernière cellule, pour le seul modèle retenu. Il ne sert jamais à comparer les modèles ni à choisir un hyperparamètre.

La boucle de validation croisée est écrite à la main plutôt qu'avec `cross_val_score`, parce que les bornes 3-sigma **et** l'imputation doivent être ré-apprises dans chaque fold. C'est le seul moyen de garantir qu'aucune information du fold de validation ne remonte — ni par l'imputation, ni par le scaler, ni par l'encodeur, ni par le filtre d'outliers.

### Catégories inconnues et `drop='first'`

`drop='first'` est **volontairement absent** du `OneHotEncoder`. Combiné à `handle_unknown='ignore'`, il encode une catégorie inconnue en vecteur nul — c'est-à-dire **exactement comme la catégorie de référence supprimée**. Un magasin jamais vu à l'entraînement serait alors prédit comme le magasin de référence, silencieusement. Le notebook en fait la démonstration :

```
Magasin inconnu encodé comme le magasin de référence ?
  AVEC drop='first'  : True   <- bug
  SANS drop='first'  : False
```

Le cas se produit réellement : les magasins **11 et 12** du hold-out sont absents du jeu d'entraînement. Un garde-fou les signale à l'exécution.

### Convergence du Lasso

La target vaut ~10⁶ $ et la tolérance de la descente par coordonnées est relative à `||y||²`. Avec `max_iter=1000` (défaut), le Lasso **ne converge pas** et ses coefficients sont invalides. Mesuré sur les 50 folds :

| `max_iter` | α=0.1 | α=1 | α=10 |
|---|---|---|---|
| 1 000 (défaut) | 50/50 échouent | 50/50 | 50/50 |
| 100 000 | 42/50 | 11/50 | 0/50 |
| **1 000 000** | 11/50 | **0/50** | **0/50** |

D'où `max_iter=1_000_000` et une grille d'`alpha` calée sur l'échelle de la target : `[1, 10, 100, 1000, 10000, 100000]`. Avec `y ~ 10⁶`, un `alpha` inférieur à 1 ne régularise rien du tout. Le `alpha` retenu vaut **10**.

---

## 🚀 Installation et utilisation

```bash
pip install -r requirements.txt
jupyter lab 01-Walmart_sales.ipynb
```

Puis **Kernel → Restart Kernel and Run All Cells**. Le notebook s'exécute de bout en bout sans erreur ni avertissement (~2 min).

### Tests

Le code réutilisable est dans `src/walmart.py`, couvert par une suite de tests :

```bash
python -m pytest -q          # 37 tests
```

Ils vérifient notamment les points où une erreur serait silencieuse : la reconstruction de `Holiday_Flag` reproduit bien toutes les valeurs connues, le filtre 3-sigma conserve les `NaN` destinés à l'imputation, l'imputation retombe sur la médiane globale pour un magasin inconnu, aucune feature n'est constante, et une catégorie inconnue reste distincte de la catégorie de référence.

Le modèle entraîné est sauvegardé en fin de notebook et s'applique directement à des données brutes :

```python
import joblib
pipe = joblib.load('models/walmart_pipeline.pkl')
pipe.predict(X)   # imputation + scaling + encodage inclus
```

### Diffs git lisibles (optionnel)

Le notebook conserve volontairement ses sorties et ses figures : c'est le livrable, il doit être lisible sans ré-exécution. Pour éviter des diffs illisibles, `.gitattributes` déclare un driver dédié, à activer une fois :

```bash
pip install nbdime && nbdime config-git --enable --global
```

---

## 📁 Structure du projet

```
jedha-walmart-project/
├── 01-Walmart_sales.ipynb          # LIVRABLE : EDA + modélisation, conforme à l'énoncé
├── 02-Walmart_ameliorations.ipynb  # Exploration hors périmètre (voir ci-dessous)
├── Walmart_Store_sales.csv         # Dataset
├── src/
│   ├── walmart.py                  # Pipeline et protocole d'évaluation du livrable
│   └── improvements.py             # Pistes hors périmètre, isolées du livrable
├── tests/                          # 37 tests (pytest)
├── requirements.txt                # Versions figées
├── .gitattributes                  # Driver de diff pour les .ipynb
├── README.md
└── models/                         # Pipeline sérialisé (non versionné, régénéré à l'exécution)
```

L'énoncé du projet est intégré au notebook (cellules markdown d'introduction).

### Notebook d'exploration (hors périmètre)

`02-Walmart_ameliorations.ipynb` teste quatre pistes qui **s'écartent de l'énoncé** — transformation `log1p` de la target, encodage par cible des magasins, modèles non linéaires, `ElasticNet` — dans le seul but de chiffrer ce que le livrable laisse sur la table. Il utilise le même découpage et le même protocole, donc les chiffres sont directement comparables.

Résultat : **une seule piste bat le livrable, mais elle le bat sur tous les critères.** `log1p(y) + Ridge(0.001)` obtient R² = 0.895 contre 0.865, avec un écart-type plus faible (0.224 contre 0.268), un meilleur pire fold (-0.56 contre -0.78), et gagne **46 folds sur 50** (p = 1.0e-11). Le coût de la conformité stricte à l'énoncé est donc chiffré : **environ 0.03 de R²**.

Les six autres pistes perdent sur la moyenne. Deux méritent d'être signalées :

- **cinq pistes sur sept sont plus régulières** que la référence, et aucune ne descend en négatif là où elle tombe à -0.78 ;
- l'encodage par cible sacrifie 0.05 de R² moyen mais **divise par trois l'erreur** sur les magasins absents du jeu d'entraînement (709 936 $ → 233 575 $).

Rien de ce notebook n'est reporté dans le livrable.

---

## 📝 Livrables

- ✅ Visualisations (EDA) — distribution, corrélations, outliers, ventes par magasin, jours fériés, résidus
- ✅ Modèle de régression linéaire
- ✅ Évaluation par métriques pertinentes (RMSE, MAE, R², R² ajusté) en validation croisée
- ✅ Interprétation des coefficients
- ✅ Modèles régularisés (Ridge et Lasso)
- ✅ Optimisation `GridSearchCV` (bonus)
- ✅ Comparaison finale des modèles
- ✅ Pipeline sérialisé, applicable à des données brutes
- ✅ Code réutilisable isolé dans `src/` et couvert par 37 tests

---

## 📚 Ressources

- [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) · [ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) · [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) · [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) · [RepeatedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html)
- [Common pitfalls in the interpretation of coefficients](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html)
