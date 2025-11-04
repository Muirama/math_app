# GUIDE DE DÉMARRAGE RAPIDE

## Installation

### Étape 1 : Vérifier Python
```powershell
python --version
```
Assurez-vous d'avoir Python 3.8 ou supérieur.

### Étape 2 : Installer les dépendances
```powershell
cd c:\Python
pip install -r requirements.txt
```

Cela installera :
- numpy (calculs numériques)
- matplotlib (graphiques)
- pulp (programmation linéaire)

### Étape 3 : Lancer l'application
```powershell
python main.py
```

## Premier test rapide

### Test 1 : Système Linéaire
1. Ouvrir l'onglet "Système Linéaire"
2. Cliquer sur "Charger exemple"
3. Sélectionner méthode "gauss"
4. Cliquer sur "Résoudre"
5. Vérifier que la solution apparaît dans la zone de droite

### Test 2 : Programmation Linéaire
1. Ouvrir l'onglet "Programmation Linéaire"
2. Les valeurs par défaut sont déjà chargées
3. Cliquer sur "Résoudre"
4. Vérifier Z optimal ≈ 33

### Test 3 : Régression Linéaire
1. Ouvrir l'onglet "Régression Linéaire"
2. Cliquer sur "Générer données aléatoires"
3. Cliquer sur "Calculer régression"
4. Un graphique devrait apparaître avec la droite de régression

### Test 4 : Chaîne de Markov
1. Ouvrir l'onglet "Processus Stochastique"
2. Sous-onglet "Chaîne de Markov"
3. Cliquer sur "Charger exemple (météo)"
4. Cliquer sur "Simuler"
5. Deux graphiques apparaissent : trajectoire et distribution

### Test 5 : Marche Aléatoire
1. Onglet "Processus Stochastique"
2. Sous-onglet "Marche Aléatoire"
3. Sélectionner "2D"
4. Cliquer sur "Simuler"
5. Une trajectoire colorée apparaît

## Résolution de problèmes

### Erreur : "No module named 'numpy'"
```powershell
pip install numpy matplotlib pulp
```

### Erreur : "No module named 'tkinter'"
Sur Windows, Tkinter est normalement inclus. Si absent :
- Réinstaller Python en cochant "tcl/tk and IDLE"

### L'application ne se lance pas
Vérifier que vous êtes dans le bon répertoire :
```powershell
cd c:\Python
python main.py
```

### Les graphiques ne s'affichent pas
- Vérifier que matplotlib est installé : `pip install matplotlib`
- Redémarrer l'application

## Utilisation avancée

### Charger un fichier CSV pour régression
1. Préparer un CSV avec 2 colonnes : x,y
2. Onglet "Régression Linéaire" → "Charger CSV"
3. Sélectionner votre fichier
4. Cliquer "Calculer régression"

Exemples fournis dans le dossier `data/` :
- regression_data.csv
- regression_data2.csv

### Créer une chaîne de Markov personnalisée
1. Choisir le nombre d'états (2 à 5)
2. Cliquer "Générer matrice"
3. Remplir les probabilités (IMPORTANT : chaque ligne doit sommer à 1.0)
4. Entrer le nombre de transitions
5. Cliquer "Simuler"

Exemple matrice 2 états (pile/face) :
```
0.5  0.5
0.5  0.5
```

## Documentation complète

- **README.md** : Documentation utilisateur complète
- **RAPPORT_TECHNIQUE.md** : Détails des algorithmes et architecture
- **AI_USAGE.txt** : Déclaration d'utilisation de l'IA

## Support

En cas de problème, vérifier :
1. Version de Python (≥ 3.8)
2. Dépendances installées (`pip list`)
3. Messages d'erreur dans la console

---

Bon calcul ! 🎓
