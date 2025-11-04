"""
Module de régression linéaire
Implémente la régression linéaire simple et multiple avec visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import csv


class LinearRegression:
    """
    Classe pour effectuer une régression linéaire
    """
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.r_squared = None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajuste le modèle de régression linéaire
        
        Args:
            X: Variables explicatives (n x m)
            y: Variable à prédire (n,)
        """
        # Ajouter une colonne de 1 pour l'intercept
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Méthode des moindres carrés : β = (X^T X)^-1 X^T y
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        
        # Stocker les données d'entraînement
        self.X_train = X
        self.y_train = y
        
        # Calculer R²
        self._calculate_r_squared()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les valeurs pour de nouvelles données
        
        Args:
            X: Nouvelles variables explicatives
        
        Returns:
            Prédictions
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        return self.intercept + np.dot(X, self.coefficients)
    
    def _calculate_r_squared(self) -> None:
        """
        Calcule le coefficient de détermination R²
        """
        y_pred = self.predict(self.X_train)
        ss_tot = np.sum((self.y_train - np.mean(self.y_train)) ** 2)
        ss_res = np.sum((self.y_train - y_pred) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
    
    def get_equation(self) -> str:
        """
        Retourne l'équation de la droite de régression
        """
        if self.coefficients is None:
            return "Modèle non ajusté"
        
        if len(self.coefficients) == 1:
            return f"y = {self.intercept:.4f} + {self.coefficients[0]:.4f}x"
        else:
            terms = [f"{self.intercept:.4f}"]
            for i, coef in enumerate(self.coefficients):
                terms.append(f"{coef:.4f}x{i+1}")
            return "y = " + " + ".join(terms)
    
    def plot_regression(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Trace le graphique de régression (uniquement pour régression simple)
        
        Args:
            save_path: Chemin pour sauvegarder le graphique
        
        Returns:
            Figure matplotlib
        """
        if len(self.coefficients) != 1:
            raise ValueError("Le tracé n'est disponible que pour la régression simple")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Points de données
        ax.scatter(self.X_train, self.y_train, alpha=0.6, label='Données')
        
        # Droite de régression
        X_line = np.linspace(self.X_train.min(), self.X_train.max(), 100)
        y_line = self.predict(X_line)
        ax.plot(X_line, y_line, 'r-', linewidth=2, label='Régression')
        
        # Annotations
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Régression Linéaire\n{self.get_equation()}\nR² = {self.r_squared:.4f}', 
                    fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def get_statistics(self) -> dict:
        """
        Retourne les statistiques du modèle
        """
        if self.coefficients is None:
            return {'error': 'Modèle non ajusté'}
        
        y_pred = self.predict(self.X_train)
        residuals = self.y_train - y_pred
        
        return {
            'equation': self.get_equation(),
            'intercept': self.intercept,
            'coefficients': self.coefficients,
            'r_squared': self.r_squared,
            'rmse': np.sqrt(np.mean(residuals ** 2)),
            'mae': np.mean(np.abs(residuals)),
            'n_samples': len(self.y_train)
        }


def load_data_from_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les données depuis un fichier CSV
    
    Args:
        filepath: Chemin vers le fichier CSV
    
    Returns:
        Tuple (X, y)
    """
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    X = data[:, 0]
    y = data[:, 1]
    return X, y


def perform_regression(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Effectue une régression linéaire complète
    
    Args:
        X: Variables explicatives
        y: Variable à prédire
    
    Returns:
        Dictionnaire avec les résultats
    """
    try:
        model = LinearRegression()
        model.fit(X, y)
        
        stats = model.get_statistics()
        stats['success'] = True
        stats['model'] = model
        
        return stats
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Fonction de test
if __name__ == "__main__":
    print("=== Test Régression Linéaire ===")
    
    # Générer des données de test
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2.5 * X + 1.0 + np.random.normal(0, 2, 50)
    
    # Effectuer la régression
    result = perform_regression(X, y)
    
    if result['success']:
        print(f"\nÉquation: {result['equation']}")
        print(f"R² = {result['r_squared']:.4f}")
        print(f"RMSE = {result['rmse']:.4f}")
        print(f"MAE = {result['mae']:.4f}")
        print(f"Nombre d'échantillons: {result['n_samples']}")
        
        # Créer le graphique
        model = result['model']
        fig = model.plot_regression()
        plt.show()
    else:
        print(f"Erreur: {result['error']}")
