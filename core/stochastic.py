"""
Module de processus stochastiques
Implémente la simulation de chaînes de Markov et marches aléatoires
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class MarkovChain:
    """
    Classe pour simuler des chaînes de Markov
    """
    
    def __init__(self, transition_matrix: np.ndarray, states: Optional[List[str]] = None):
        """
        Initialise une chaîne de Markov
        
        Args:
            transition_matrix: Matrice de transition (n x n)
            states: Noms des états (optionnel)
        """
        self.transition_matrix = np.array(transition_matrix, dtype=float)
        self.n_states = self.transition_matrix.shape[0]
        
        # Vérifier que c'est une matrice stochastique
        if not self._is_stochastic():
            raise ValueError("La matrice de transition n'est pas stochastique (somme des lignes != 1)")
        
        # Noms des états
        if states is None:
            self.states = [f"État {i}" for i in range(self.n_states)]
        else:
            self.states = states
        
        self.current_state = None
        self.history = []
    
    def _is_stochastic(self) -> bool:
        """
        Vérifie si la matrice est stochastique
        """
        row_sums = np.sum(self.transition_matrix, axis=1)
        return np.allclose(row_sums, 1.0)
    
    def simulate(self, n_steps: int, initial_state: int = 0) -> List[int]:
        """
        Simule la chaîne de Markov
        
        Args:
            n_steps: Nombre de transitions
            initial_state: État initial
        
        Returns:
            Liste des états visités
        """
        self.current_state = initial_state
        self.history = [initial_state]
        
        for _ in range(n_steps):
            # Probabilités de transition depuis l'état actuel
            transition_probs = self.transition_matrix[self.current_state]
            
            # Choisir le prochain état selon ces probabilités
            next_state = np.random.choice(self.n_states, p=transition_probs)
            
            self.history.append(next_state)
            self.current_state = next_state
        
        return self.history
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Calcule la distribution stationnaire (si elle existe)
        
        Returns:
            Vecteur de probabilités stationnaires
        """
        # Méthode: résoudre πP = π avec Σπ_i = 1
        # Équivalent à: (P^T - I)π = 0 avec contrainte Σπ_i = 1
        
        n = self.n_states
        A = np.transpose(self.transition_matrix) - np.eye(n)
        # Remplacer la dernière équation par la contrainte de somme
        A[-1, :] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1
        
        try:
            stationary = np.linalg.solve(A, b)
            return stationary
        except:
            # Si pas de solution unique, utiliser la méthode des puissances
            P = self.transition_matrix
            for _ in range(1000):
                P = np.dot(P, self.transition_matrix)
            return P[0, :]
    
    def plot_trajectory(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Trace la trajectoire de la chaîne
        
        Args:
            save_path: Chemin pour sauvegarder le graphique
        
        Returns:
            Figure matplotlib
        """
        if not self.history:
            raise ValueError("Aucune simulation effectuée")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Graphique 1: Trajectoire
        ax1.plot(self.history, 'o-', markersize=4, linewidth=1)
        ax1.set_xlabel('Temps', fontsize=12)
        ax1.set_ylabel('État', fontsize=12)
        ax1.set_title('Trajectoire de la chaîne de Markov', fontsize=14)
        ax1.set_yticks(range(self.n_states))
        ax1.set_yticklabels(self.states)
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Fréquence des états
        state_counts = np.bincount(self.history, minlength=self.n_states)
        state_frequencies = state_counts / len(self.history)
        
        x_pos = np.arange(self.n_states)
        ax2.bar(x_pos, state_frequencies, alpha=0.7, label='Fréquence observée')
        
        # Ajouter la distribution stationnaire si disponible
        try:
            stationary = self.get_stationary_distribution()
            ax2.plot(x_pos, stationary, 'ro-', linewidth=2, markersize=8, 
                    label='Distribution stationnaire')
        except:
            pass
        
        ax2.set_xlabel('État', fontsize=12)
        ax2.set_ylabel('Fréquence', fontsize=12)
        ax2.set_title('Distribution des états', fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.states)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def get_statistics(self) -> dict:
        """
        Retourne les statistiques de la simulation
        """
        if not self.history:
            return {'error': 'Aucune simulation effectuée'}
        
        state_counts = np.bincount(self.history, minlength=self.n_states)
        state_frequencies = state_counts / len(self.history)
        
        try:
            stationary = self.get_stationary_distribution()
        except:
            stationary = None
        
        return {
            'n_steps': len(self.history) - 1,
            'initial_state': self.history[0],
            'final_state': self.history[-1],
            'state_frequencies': state_frequencies,
            'stationary_distribution': stationary,
            'unique_states_visited': len(np.unique(self.history))
        }


class RandomWalk:
    """
    Classe pour simuler des marches aléatoires
    """
    
    def __init__(self, dimension: int = 1):
        """
        Initialise une marche aléatoire
        
        Args:
            dimension: Dimension de l'espace (1D ou 2D)
        """
        self.dimension = dimension
        self.trajectory = []
    
    def simulate(self, n_steps: int, step_size: float = 1.0) -> np.ndarray:
        """
        Simule une marche aléatoire
        
        Args:
            n_steps: Nombre de pas
            step_size: Taille du pas
        
        Returns:
            Trajectoire
        """
        if self.dimension == 1:
            # Marche aléatoire 1D: chaque pas est +1 ou -1
            steps = np.random.choice([-step_size, step_size], size=n_steps)
            positions = np.cumsum(np.concatenate([[0], steps]))
            self.trajectory = positions
        
        elif self.dimension == 2:
            # Marche aléatoire 2D: angles aléatoires
            angles = np.random.uniform(0, 2*np.pi, n_steps)
            dx = step_size * np.cos(angles)
            dy = step_size * np.sin(angles)
            
            x = np.cumsum(np.concatenate([[0], dx]))
            y = np.cumsum(np.concatenate([[0], dy]))
            
            self.trajectory = np.column_stack([x, y])
        
        else:
            raise ValueError("Seules les dimensions 1 et 2 sont supportées")
        
        return self.trajectory
    
    def plot_walk(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Trace la marche aléatoire
        
        Args:
            save_path: Chemin pour sauvegarder le graphique
        
        Returns:
            Figure matplotlib
        """
        if len(self.trajectory) == 0:
            raise ValueError("Aucune simulation effectuée")
        
        fig = plt.figure(figsize=(10, 6))
        
        if self.dimension == 1:
            ax = fig.add_subplot(111)
            ax.plot(self.trajectory, linewidth=1.5)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Temps', fontsize=12)
            ax.set_ylabel('Position', fontsize=12)
            ax.set_title('Marche Aléatoire 1D', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        elif self.dimension == 2:
            ax = fig.add_subplot(111)
            x = self.trajectory[:, 0]
            y = self.trajectory[:, 1]
            
            # Tracer la trajectoire avec gradient de couleur
            points = ax.scatter(x, y, c=range(len(x)), cmap='viridis', 
                              s=20, alpha=0.6)
            ax.plot(x, y, 'b-', alpha=0.3, linewidth=0.5)
            
            # Marquer le début et la fin
            ax.plot(x[0], y[0], 'go', markersize=12, label='Début')
            ax.plot(x[-1], y[-1], 'ro', markersize=12, label='Fin')
            
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_title('Marche Aléatoire 2D', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            plt.colorbar(points, label='Temps')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# Fonction de test
if __name__ == "__main__":
    print("=== Test Chaîne de Markov ===")
    
    # Exemple: météo (Soleil, Nuageux, Pluie)
    transition = np.array([
        [0.7, 0.2, 0.1],  # Depuis Soleil
        [0.3, 0.4, 0.3],  # Depuis Nuageux
        [0.2, 0.3, 0.5]   # Depuis Pluie
    ])
    
    states = ['Soleil', 'Nuageux', 'Pluie']
    
    markov = MarkovChain(transition, states)
    history = markov.simulate(100, initial_state=0)
    
    stats = markov.get_statistics()
    print(f"\nNombre de transitions: {stats['n_steps']}")
    print(f"État initial: {states[stats['initial_state']]}")
    print(f"État final: {states[stats['final_state']]}")
    print("\nFréquences observées:")
    for i, freq in enumerate(stats['state_frequencies']):
        print(f"  {states[i]}: {freq:.3f}")
    
    if stats['stationary_distribution'] is not None:
        print("\nDistribution stationnaire:")
        for i, prob in enumerate(stats['stationary_distribution']):
            print(f"  {states[i]}: {prob:.3f}")
    
    print("\n=== Test Marche Aléatoire ===")
    walk = RandomWalk(dimension=2)
    walk.simulate(1000)
    print(f"Position finale: {walk.trajectory[-1]}")
    print(f"Distance à l'origine: {np.linalg.norm(walk.trajectory[-1]):.2f}")
