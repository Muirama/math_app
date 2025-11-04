"""
Module de programmation linéaire
Utilise PuLP pour résoudre des problèmes d'optimisation linéaire
"""

import pulp
import numpy as np
from typing import Dict, List, Tuple, Optional


class LinearProgrammingSolver:
    """
    Classe pour résoudre des problèmes de programmation linéaire
    """
    
    def __init__(self):
        self.problem = None
        self.variables = {}
        self.result = None
    
    def solve_standard_problem(self, 
                              objective_coeffs: Dict[str, float],
                              constraints: List[Tuple[Dict[str, float], str, float]],
                              sense: str = 'maximize',
                              bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> dict:
        """
        Résout un problème de programmation linéaire standard
        
        Args:
            objective_coeffs: Coefficients de la fonction objectif {variable: coefficient}
            constraints: Liste de contraintes [(coeffs, comparateur, valeur)]
                        comparateur peut être '<=', '>=', ou '=='
            sense: 'maximize' ou 'minimize'
            bounds: Bornes des variables {variable: (min, max)}
        
        Returns:
            Dictionnaire avec les résultats
        """
        try:
            # Créer le problème
            if sense == 'maximize':
                self.problem = pulp.LpProblem("Problem", pulp.LpMaximize)
            else:
                self.problem = pulp.LpProblem("Problem", pulp.LpMinimize)
            
            # Créer les variables de décision
            self.variables = {}
            for var_name in objective_coeffs.keys():
                if bounds and var_name in bounds:
                    low, up = bounds[var_name]
                    self.variables[var_name] = pulp.LpVariable(var_name, lowBound=low, upBound=up)
                else:
                    # Par défaut, variables non-négatives
                    self.variables[var_name] = pulp.LpVariable(var_name, lowBound=0)
            
            # Définir la fonction objectif
            objective = pulp.lpSum([objective_coeffs[var] * self.variables[var] 
                                   for var in objective_coeffs.keys()])
            self.problem += objective
            
            # Ajouter les contraintes
            for i, (coeffs, comparator, value) in enumerate(constraints):
                constraint_expr = pulp.lpSum([coeffs[var] * self.variables[var] 
                                             for var in coeffs.keys()])
                
                if comparator == '<=':
                    self.problem += constraint_expr <= value, f"Constraint_{i}"
                elif comparator == '>=':
                    self.problem += constraint_expr >= value, f"Constraint_{i}"
                elif comparator == '==':
                    self.problem += constraint_expr == value, f"Constraint_{i}"
                else:
                    raise ValueError(f"Comparateur invalide: {comparator}")
            
            # Résoudre
            status = self.problem.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Extraire les résultats
            if status == pulp.LpStatusOptimal:
                solution = {var: self.variables[var].varValue for var in self.variables}
                optimal_value = pulp.value(self.problem.objective)
                
                return {
                    'success': True,
                    'status': 'Optimal',
                    'objective_value': optimal_value,
                    'solution': solution,
                    'sense': sense
                }
            else:
                return {
                    'success': False,
                    'status': pulp.LpStatus[status],
                    'message': 'Aucune solution optimale trouvée'
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def solve_simple_2d(self, 
                       c1: float, c2: float,
                       constraints_matrix: np.ndarray,
                       constraints_rhs: np.ndarray,
                       sense: str = 'maximize') -> dict:
        """
        Résout un problème simple à 2 variables (pour visualisation graphique)
        
        Args:
            c1, c2: Coefficients de la fonction objectif (Z = c1*x + c2*y)
            constraints_matrix: Matrice des contraintes (n x 2)
            constraints_rhs: Second membre des contraintes (n x 1)
            sense: 'maximize' ou 'minimize'
        
        Returns:
            Dictionnaire avec les résultats
        """
        # Créer les dictionnaires pour la méthode standard
        objective_coeffs = {'x': c1, 'y': c2}
        
        constraints = []
        for i in range(len(constraints_rhs)):
            coeffs = {'x': constraints_matrix[i, 0], 'y': constraints_matrix[i, 1]}
            constraints.append((coeffs, '<=', constraints_rhs[i]))
        
        return self.solve_standard_problem(objective_coeffs, constraints, sense)
    
    def get_problem_string(self) -> str:
        """
        Retourne une représentation textuelle du problème
        """
        if self.problem:
            return str(self.problem)
        return "Aucun problème défini"


def solve_example_problem():
    """
    Exemple : Maximiser Z = 3x + 2y
    Contraintes:
        2x + y <= 18
        2x + 3y <= 42
        3x + y <= 24
        x, y >= 0
    """
    solver = LinearProgrammingSolver()
    
    objective = {'x': 3, 'y': 2}
    constraints = [
        ({'x': 2, 'y': 1}, '<=', 18),
        ({'x': 2, 'y': 3}, '<=', 42),
        ({'x': 3, 'y': 1}, '<=', 24)
    ]
    
    result = solver.solve_standard_problem(objective, constraints, 'maximize')
    return result


# Fonction de test
if __name__ == "__main__":
    print("=== Test Programmation Linéaire ===")
    print("\nProblème: Maximiser Z = 3x + 2y")
    print("Contraintes:")
    print("  2x + y <= 18")
    print("  2x + 3y <= 42")
    print("  3x + y <= 24")
    print("  x, y >= 0")
    
    result = solve_example_problem()
    
    if result['success']:
        print(f"\nStatut: {result['status']}")
        print(f"Valeur optimale: Z = {result['objective_value']:.2f}")
        print("Solution:")
        for var, value in result['solution'].items():
            print(f"  {var} = {value:.2f}")
    else:
        print(f"\nErreur: {result.get('error', result.get('message'))}")
