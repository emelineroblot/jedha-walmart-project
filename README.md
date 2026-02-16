# 🛒 Walmart Sales Prediction

Projet de **Machine Learning supervisé** pour prédire les ventes hebdomadaires des magasins Walmart en utilisant des indicateurs économiques.

---

## 📋 Description du projet

Walmart souhaite construire un modèle de Machine Learning capable d'estimer les ventes hebdomadaires dans ses magasins avec la meilleure précision possible. Ce modèle aide à :
- Comprendre comment les ventes sont influencées par les indicateurs économiques
- Planifier les futures campagnes marketing

---

## 🎯 Objectifs

Le projet est divisé en trois parties :

1. **Partie 1** : Analyse exploratoire (EDA) et preprocessing des données
2. **Partie 2** : Entraîner un modèle de **régression linéaire** (baseline)
3. **Partie 3** : Réduire l'overfitting avec des modèles **régularisés** (Ridge et Lasso)

---

## 📊 Dataset

Le dataset contient **150 lignes** et **8 colonnes** :

| Colonne | Type | Description |
|---------|------|-------------|
| `Store` | Catégorielle | Identifiant du magasin (1-20) |
| `Date` | Date | Date de la semaine |
| `Weekly_Sales` | Numérique | **Target** - Ventes hebdomadaires en $ |
| `Holiday_Flag` | Catégorielle | Indicateur jour férié (0=Non, 1=Oui) |
| `Temperature` | Numérique | Température moyenne (°F) |
| `Fuel_Price` | Numérique | Prix du carburant ($) |
| `CPI` | Numérique | Indice des prix à la consommation |
| `Unemployment` | Numérique | Taux de chômage (%) |

**Après nettoyage** : 71 lignes exploitables (suppression des NaN et outliers).

---

## 🛠️ Étapes du projet

### Partie 1 : EDA et Preprocessing

#### 1. Exploration des données
- Chargement du dataset
- Statistiques descriptives (`.info()`, `.describe()`)
- Analyse des valeurs manquantes (~8-12% par colonne)
- Visualisations :
  - Distribution de `Weekly_Sales`
  - Matrice de corrélation
  - Boxplots pour détecter les outliers
  - Ventes moyennes par magasin
  - Impact des jours fériés sur les ventes

#### 2. Nettoyage des données
- Suppression des lignes où `Weekly_Sales` est NaN (14 lignes)
- Transformation de la colonne `Date` en 4 features numériques :
  - `Year`
  - `Month`
  - `Day`
  - `DayOfWeek` (0=Lundi, 6=Dimanche)
- Suppression des lignes avec NaN restants
- **Suppression des outliers** avec la règle des 3-sigma (mean ± 3×std) sur :
  - `Temperature`
  - `Fuel_Price`
  - `CPI`
  - `Unemployment`
- **Dataset final** : 71 lignes

#### 3. Préparation pour le ML
- Séparation X (features) et y (target)
- Identification des types de variables :
  - **Catégorielles** : `Store`, `Holiday_Flag`
  - **Numériques** : `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `Year`, `Month`, `Day`, `DayOfWeek`
- Split train/test : **80/20** (56 train, 15 test)
- Preprocessing avec `ColumnTransformer` :
  - `StandardScaler` pour les variables numériques
  - `OneHotEncoder(handle_unknown='ignore')` pour les catégorielles

---

### Partie 2 : Régression Linéaire (Baseline)

#### 1. Entraînement
- Modèle : `LinearRegression()`
- Prédictions sur train et test

#### 2. Évaluation
Métriques utilisées :
- **RMSE** (Root Mean Squared Error) : erreur moyenne en dollars
- **MAE** (Mean Absolute Error) : erreur absolue moyenne
- **R²** (Coefficient de détermination) : variance expliquée (0 à 1)

#### 3. Interprétation des coefficients
- Extraction des coefficients avec `.coef_`
- Identification des features les plus importantes
- Visualisation avec barplot horizontal

---

### Partie 3 : Régularisation (Ridge et Lasso)

#### 1. Ridge Regression
- Modèle : `Ridge(alpha=1.0)`
- Pénalise les gros coefficients pour réduire l'overfitting
- Évaluation sur train et test

#### 2. Lasso Regression
- Modèle : `Lasso(alpha=1.0)`
- Peut mettre certains coefficients à zéro (sélection de features automatique)
- Affichage des features éliminées

#### 3. GridSearchCV (Bonus)
- Optimisation du paramètre `alpha` par validation croisée (5 folds)
- **Ridge** : test de 7 valeurs [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- **Lasso** : test de 6 valeurs [0.001, 0.01, 0.1, 1, 10, 100]
- Sélection automatique du meilleur `alpha`

#### 4. Comparaison finale
Tableau comparatif des 5 modèles :
1. Linear Regression
2. Ridge (alpha=1)
3. Lasso (alpha=1)
4. Ridge optimisé (GridSearch)
5. Lasso optimisé (GridSearch)

Visualisation comparative des R² et RMSE.

---

## 🚀 Installation et utilisation

### Prérequis
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Lancer le projet
```bash
cd walmart_project
jupyter notebook 01-Walmart_sales.ipynb
```

### Exécution
Dans Jupyter :
1. **Kernel > Restart & Run All** pour exécuter toutes les cellules
2. Les résultats s'affichent séquentiellement avec visualisations

---

## 📈 Résultats attendus

Les modèles régularisés (Ridge/Lasso) devraient :
- Réduire l'écart entre les performances train et test
- Améliorer la généralisation
- Obtenir un R² entre 0.6 et 0.9 selon la qualité des données

Les features importantes identifiées devraient inclure :
- L'identifiant du magasin (`Store`)
- Les indicateurs économiques (CPI, Unemployment)
- Les caractéristiques temporelles (Month, Year)

---

## 📁 Structure du projet

```
walmart_project/
│
├── Walmart_Store_sales.csv          # Dataset
├── 01-Walmart_sales.ipynb            # Notebook principal
├── Contexte_projet.txt               # Brief du projet
└── README.md                         # Documentation
```

---

## 🔍 Points clés techniques

### Gestion des catégories inconnues
```python
OneHotEncoder(handle_unknown='ignore')
```
Avec un petit dataset (71 lignes) et 20 magasins, certains stores peuvent n'apparaître que dans le test set. Le paramètre `handle_unknown='ignore'` évite les erreurs en créant des vecteurs de zéros.

### Règle des 3-sigma
```python
lower_bound = mean - 3 * std
upper_bound = mean + 3 * std
```
Les valeurs hors de cet intervalle sont considérées comme outliers (99.7% des données normales sont dans cet intervalle).

### Régularisation
- **Ridge (L2)** : pénalise la somme des carrés des coefficients → réduit leur magnitude
- **Lasso (L1)** : pénalise la somme des valeurs absolues → peut mettre des coefficients exactement à 0

---

## 📚 Ressources

- [Scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Scikit-learn Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

---

## 📝 Livrables

✅ Visualisations (EDA)
✅ Modèle de régression linéaire
✅ Évaluation avec métriques pertinentes (RMSE, MAE, R²)
✅ Interprétation des coefficients
✅ Modèles régularisés (Ridge et Lasso)
✅ Optimisation GridSearchCV (bonus)
✅ Comparaison finale des modèles
