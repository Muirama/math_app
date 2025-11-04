# Application de Modélisation Mathématique

## Description

Application Python avec interface graphique permettant d'effectuer des calculs mathématiques avancés dans les domaines suivants :

1. **Résolution de systèmes linéaires** - Méthodes de Gauss et décomposition LU
2. **Programmation linéaire** - Optimisation avec contraintes
3. **Régression linéaire** - Ajustement de modèles aux données
4. **Processus stochastiques** - Chaînes de Markov et marches aléatoires

## Structure du Projet

```
Python/
├── main.py                 # Point d'entrée de l'application
├── requirements.txt        # Dépendances Python
├── README.md              # Ce fichier
├── AI_USAGE.txt           # Documentation de l'utilisation de l'IA
├── core/                  # Modules de calcul
│   ├── __init__.py
│   ├── linear_system.py       # Résolution de systèmes linéaires
│   ├── linear_programming.py  # Programmation linéaire
│   ├── regression.py          # Régression linéaire
│   └── stochastic.py          # Processus stochastiques
├── ui/                    # Interface graphique
│   ├── __init__.py
│   └── main_window.py         # Interface Tkinter principale
└── data/                  # Fichiers de données de test
    ├── regression_data.csv
    └── regression_data2.csv
```

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
cd Python
pip install -r requirements.txt
```

Les bibliothèques nécessaires sont :
- **NumPy** : Calculs numériques et algèbre linéaire
- **Matplotlib** : Visualisation graphique
- **PuLP** : Résolution de problèmes de programmation linéaire

## Utilisation

### Lancement de l'application

```bash
python main.py
```

### Modules disponibles

#### 1. Système Linéaire

Résout des systèmes d'équations linéaires de la forme AX = b

**Fonctionnalités :**
- Saisie manuelle de matrices jusqu'à 10×10
- Trois méthodes de résolution :
  - Élimination de Gauss (avec pivot partiel)
  - Décomposition LU
  - Solveur NumPy optimisé
- Affichage de la solution, du résidu, déterminant et nombre de condition
- Exemple pré-chargé disponible

**Exemple d'utilisation :**
1. Sélectionner la taille de la matrice (ex: 3×3)
2. Cliquer sur "Charger exemple" ou saisir vos valeurs
3. Choisir la méthode de résolution
4. Cliquer sur "Résoudre"

#### 2. Programmation Linéaire

Maximise ou minimise une fonction objectif avec contraintes linéaires

**Fonctionnalités :**
- Définition de la fonction objectif (maximiser/minimiser)
- Saisie de contraintes au format texte
- Résolution automatique avec PuLP
- Affichage de la solution optimale

**Format des contraintes :**
```
2*x + 1*y <= 18
2*x + 3*y <= 42
3*x + 1*y <= 24
```

**Exemple :**
```
Maximiser Z = 3x + 2y
sous contraintes :
  2x + y ≤ 18
  2x + 3y ≤ 42
  3x + y ≤ 24
  x, y ≥ 0
```

#### 3. Régression Linéaire

Ajuste un modèle linéaire à des données

**Fonctionnalités :**
- Chargement de données depuis CSV (2 colonnes : x, y)
- Génération de données aléatoires pour test
- Calcul de la régression par moindres carrés
- Graphique avec droite de régression
- Statistiques : R², RMSE, MAE, équation

**Format CSV :**
```csv
x,y
1.0,3.2
2.0,5.1
3.0,7.3
...
```

#### 4. Processus Stochastique

Simule des processus aléatoires

**a) Chaîne de Markov**
- Définition d'une matrice de transition stochastique
- Simulation de trajectoires
- Calcul de la distribution stationnaire
- Visualisation de la trajectoire et des fréquences
- Exemple pré-chargé (modèle météo)

**b) Marche Aléatoire**
- Marche aléatoire 1D ou 2D
- Nombre de pas et taille configurable
- Visualisation avec gradient temporel

## Tests

### Tests manuels suggérés

#### Système Linéaire
```
Matrice A:        Vecteur b:
[3   2  -1]       [1]
[2  -2   4]       [-2]
[-1 0.5 -1]       [0]

Solution attendue: x₁ ≈ 1, x₂ ≈ -2, x₃ ≈ -2
```

#### Programmation Linéaire
```
Maximiser Z = 3x + 2y
Contraintes:
  2x + y ≤ 18
  2x + 3y ≤ 42
  3x + y ≤ 24

Solution attendue: x = 3, y = 12, Z = 33
```

#### Régression Linéaire
- Utiliser les fichiers CSV dans le dossier `data/`
- Ou générer des données aléatoires

#### Chaîne de Markov
```
Matrice de transition (météo):
       Soleil  Nuageux  Pluie
Soleil   0.7     0.2     0.1
Nuageux  0.3     0.4     0.3
Pluie    0.2     0.3     0.5

Simuler 100+ transitions
```

## Algorithmes Implémentés

### 1. Élimination de Gauss
- Pivot partiel pour stabilité numérique
- Complexité : O(n³)
- Substitution arrière

### 2. Décomposition LU
- Factorisation A = LU
- Résolution en deux étapes (Ly = b puis Ux = y)

### 3. Régression Linéaire
- Méthode des moindres carrés : β = (XᵀX)⁻¹Xᵀy
- Calcul manuel du R²

### 4. Programmation Linéaire
- Utilise le solveur CBC via PuLP
- Méthode du simplexe

### 5. Chaîne de Markov
- Simulation par échantillonnage selon les probabilités de transition
- Distribution stationnaire : résolution de πP = π

### 6. Marche Aléatoire
- 1D : pas aléatoires ±1
- 2D : direction aléatoire uniforme

## Limites et Améliorations Possibles

### Limites actuelles

1. **Système linéaire** : Matrices mal conditionnées peuvent donner des résultats imprécis
2. **Programmation linéaire** : Interface limitée à 2 variables pour simplification
3. **Régression** : Uniquement régression simple (1 variable explicative)
4. **Processus stochastiques** : Pas de validation approfondie de convergence

### Améliorations futures

1. Support de la régression multiple (plusieurs variables)
2. Visualisation 3D pour programmation linéaire
3. Export des résultats en PDF
4. Analyse de sensibilité pour l'optimisation
5. Plus de processus stochastiques (processus de Poisson, etc.)
6. Gestion d'erreurs plus robuste
7. Tests unitaires automatisés
8. Sauvegarde/chargement de sessions

## Dépannage

### Erreur d'importation
```bash
# Vérifier que toutes les dépendances sont installées
pip install -r requirements.txt
```

### Problème d'affichage graphique
- Sur certains systèmes, Tkinter peut nécessiter une installation séparée
- Sous Linux : `sudo apt-get install python3-tk`

### PuLP ne trouve pas de solveur
- PuLP inclut le solveur CBC par défaut
- Réinstaller si nécessaire : `pip install --upgrade pulp`

## Auteur

Projet développé dans le cadre d'un cours de Python scientifique.

## Licence

Projet éducatif - Usage libre pour apprentissage.
