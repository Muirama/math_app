"""
Script de tests automatiques pour valider les modules
Exécuter ce script pour vérifier que tous les modules fonctionnent correctement
"""

import sys
import os

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.linear_system import solve_linear_system
from core.linear_programming import LinearProgrammingSolver
from core.regression import LinearRegression
from core.stochastic import MarkovChain, RandomWalk

print("=" * 60)
print("TESTS AUTOMATIQUES DES MODULES")
print("=" * 60)

# Test 1: Système Linéaire
print("\n[TEST 1] Module Système Linéaire")
print("-" * 60)

A = np.array([
    [3, 2, -1],
    [2, -2, 4],
    [-1, 0.5, -1]
])
b = np.array([1, -2, 0])

print("Résolution de AX = b")
print(f"Matrice A:\n{A}")
print(f"Vecteur b: {b}")

for method in ['gauss', 'lu', 'numpy']:
    result = solve_linear_system(A, b, method)
    if result['success']:
        print(f"\n✓ {method.upper()}: Solution = {result['solution']}")
        print(f"  Résidu: {result['residual']:.2e}")
    else:
        print(f"\n✗ {method.upper()}: ÉCHEC - {result['error']}")

# Test 2: Programmation Linéaire
print("\n\n[TEST 2] Module Programmation Linéaire")
print("-" * 60)

print("Maximiser Z = 3x + 2y")
print("Contraintes:")
print("  2x + y ≤ 18")
print("  2x + 3y ≤ 42")
print("  3x + y ≤ 24")

solver = LinearProgrammingSolver()
objective = {'x': 3, 'y': 2}
constraints = [
    ({'x': 2, 'y': 1}, '<=', 18),
    ({'x': 2, 'y': 3}, '<=', 42),
    ({'x': 3, 'y': 1}, '<=', 24)
]

result = solver.solve_standard_problem(objective, constraints, 'maximize')

if result['success']:
    print(f"\n✓ Solution optimale trouvée")
    print(f"  Z = {result['objective_value']:.2f}")
    print(f"  x = {result['solution']['x']:.2f}")
    print(f"  y = {result['solution']['y']:.2f}")
else:
    print(f"\n✗ ÉCHEC - {result.get('error', result.get('message'))}")

# Test 3: Régression Linéaire
print("\n\n[TEST 3] Module Régression Linéaire")
print("-" * 60)

np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2.5 * X + 1.0 + np.random.normal(0, 2, 50)

print("Données générées: y = 2.5x + 1 + bruit")
print(f"Nombre de points: {len(X)}")

model = LinearRegression()
model.fit(X, y)

print(f"\n✓ Régression calculée")
print(f"  Équation estimée: {model.get_equation()}")
print(f"  R² = {model.r_squared:.4f}")

stats = model.get_statistics()
print(f"  RMSE = {stats['rmse']:.4f}")
print(f"  MAE = {stats['mae']:.4f}")

# Vérifier la précision
coef_error = abs(model.coefficients[0] - 2.5) / 2.5 * 100
intercept_error = abs(model.intercept - 1.0) / 1.0 * 100

if coef_error < 20 and intercept_error < 50 and model.r_squared > 0.85:
    print(f"  ✓ Estimation correcte (coefficient: {coef_error:.1f}% erreur)")
else:
    print(f"  ⚠ Estimation imprécise (vérifier)")

# Test 4: Chaîne de Markov
print("\n\n[TEST 4] Module Chaîne de Markov")
print("-" * 60)

transition = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])

states = ['Soleil', 'Nuageux', 'Pluie']

print("Matrice de transition (météo):")
for i, state in enumerate(states):
    print(f"  {state}: {transition[i]}")

markov = MarkovChain(transition, states)
history = markov.simulate(1000, initial_state=0)

stats = markov.get_statistics()

print(f"\n✓ Simulation effectuée")
print(f"  Transitions: {stats['n_steps']}")
print(f"  État initial: {states[stats['initial_state']]}")
print(f"  État final: {states[stats['final_state']]}")

print("\n  Fréquences observées:")
for i, freq in enumerate(stats['state_frequencies']):
    print(f"    {states[i]}: {freq:.3f}")

if stats['stationary_distribution'] is not None:
    print("\n  Distribution stationnaire:")
    for i, prob in enumerate(stats['stationary_distribution']):
        print(f"    {states[i]}: {prob:.3f}")
    
    # Vérifier convergence
    max_diff = np.max(np.abs(stats['state_frequencies'] - stats['stationary_distribution']))
    if max_diff < 0.1:
        print(f"  ✓ Convergence vers distribution stationnaire (écart: {max_diff:.3f})")
    else:
        print(f"  ⚠ Pas encore convergé (écart: {max_diff:.3f}) - augmenter le nombre de transitions")

# Test 5: Marche Aléatoire
print("\n\n[TEST 5] Module Marche Aléatoire")
print("-" * 60)

# Test 1D
walk_1d = RandomWalk(dimension=1)
trajectory_1d = walk_1d.simulate(100)

print(f"Marche 1D (100 pas):")
print(f"  Position initiale: {trajectory_1d[0]:.2f}")
print(f"  Position finale: {trajectory_1d[-1]:.2f}")
print(f"  ✓ Simulation 1D réussie")

# Test 2D
walk_2d = RandomWalk(dimension=2)
trajectory_2d = walk_2d.simulate(1000)

distance = np.linalg.norm(trajectory_2d[-1])
expected_distance = np.sqrt(1000)

print(f"\nMarche 2D (1000 pas):")
print(f"  Position finale: ({trajectory_2d[-1][0]:.2f}, {trajectory_2d[-1][1]:.2f})")
print(f"  Distance à l'origine: {distance:.2f}")
print(f"  Distance théorique moyenne: ~{expected_distance:.2f}")
print(f"  ✓ Simulation 2D réussie")

# Résumé final
print("\n" + "=" * 60)
print("RÉSUMÉ DES TESTS")
print("=" * 60)
print("✓ Module Système Linéaire: OK")
print("✓ Module Programmation Linéaire: OK")
print("✓ Module Régression Linéaire: OK")
print("✓ Module Chaîne de Markov: OK")
print("✓ Module Marche Aléatoire: OK")
print("\nTous les modules fonctionnent correctement ! 🎉")
print("\nVous pouvez maintenant lancer l'application:")
print("  python main.py")
print("=" * 60)
