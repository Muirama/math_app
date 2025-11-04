"""
Module de résolution de systèmes linéaires
Implémente les méthodes de Gauss et de décomposition LU
"""

import numpy as np
from typing import Tuple, Optional


def gauss_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Résout le système AX = b par la méthode d'élimination de Gauss
    
    Args:
        A: Matrice des coefficients (n x n)
        b: Vecteur du second membre (n x 1)
    
    Returns:
        Vecteur solution X
    
    Raises:
        ValueError: Si la matrice est singulière
    """
    n = len(b)
    # Créer la matrice augmentée [A|b]
    Ab = np.column_stack([A.astype(float), b.astype(float)])
    
    # Phase d'élimination (triangularisation)
    for i in range(n):
        # Recherche du pivot maximal (pivot partiel)
        max_row = i + np.argmax(np.abs(Ab[i:n, i]))
        if abs(Ab[max_row, i]) < 1e-10:
            raise ValueError("La matrice est singulière ou presque singulière")
        
        # Échanger les lignes si nécessaire
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Élimination
        for j in range(i + 1, n):
            if Ab[i, i] != 0:
                factor = Ab[j, i] / Ab[i, i]
                Ab[j, i:] -= factor * Ab[i, i:]
    
    # Phase de substitution arrière
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x


def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Effectue la décomposition LU de la matrice A
    
    Args:
        A: Matrice carrée (n x n)
    
    Returns:
        Tuple (L, U) où L est triangulaire inférieure et U triangulaire supérieure
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for i in range(n):
        for j in range(i + 1, n):
            if U[i, i] != 0:
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j, i:] -= factor * U[i, i:]
    
    return L, U


def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Résout le système AX = b en utilisant la décomposition LU
    
    Args:
        A: Matrice des coefficients (n x n)
        b: Vecteur du second membre (n x 1)
    
    Returns:
        Vecteur solution X
    """
    L, U = lu_decomposition(A)
    n = len(b)
    
    # Résolution de Ly = b (substitution avant)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Résolution de Ux = y (substitution arrière)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x


def solve_linear_system(A: np.ndarray, b: np.ndarray, method: str = 'gauss') -> dict:
    """
    Résout un système linéaire avec la méthode choisie
    
    Args:
        A: Matrice des coefficients
        b: Vecteur du second membre
        method: 'gauss', 'lu' ou 'numpy'
    
    Returns:
        Dictionnaire contenant la solution et des informations supplémentaires
    """
    try:
        # Vérifications de base
        if A.shape[0] != A.shape[1]:
            raise ValueError("La matrice A doit être carrée")
        if A.shape[0] != len(b):
            raise ValueError("Les dimensions de A et b ne correspondent pas")
        
        # Résolution selon la méthode choisie
        if method == 'gauss':
            solution = gauss_elimination(A, b)
            method_used = "Élimination de Gauss"
        elif method == 'lu':
            solution = solve_lu(A, b)
            method_used = "Décomposition LU"
        elif method == 'numpy':
            solution = np.linalg.solve(A, b)
            method_used = "NumPy (solver optimisé)"
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        # Vérification de la solution
        residual = np.linalg.norm(np.dot(A, solution) - b)
        
        return {
            'success': True,
            'solution': solution,
            'method': method_used,
            'residual': residual,
            'condition_number': np.linalg.cond(A),
            'determinant': np.linalg.det(A)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'solution': None
        }


# Fonction de test
if __name__ == "__main__":
    # Test avec un système 3x3
    A = np.array([
        [3, 2, -1],
        [2, -2, 4],
        [-1, 0.5, -1]
    ])
    b = np.array([1, -2, 0])
    
    print("=== Test Système Linéaire ===")
    print(f"Matrice A:\n{A}")
    print(f"Vecteur b: {b}")
    
    for method in ['gauss', 'lu', 'numpy']:
        result = solve_linear_system(A, b, method)
        if result['success']:
            print(f"\n{result['method']}:")
            print(f"Solution: {result['solution']}")
            print(f"Résidu: {result['residual']:.2e}")
        else:
            print(f"\nErreur avec {method}: {result['error']}")
