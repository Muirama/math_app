"""
Interface graphique principale de l'application
Utilise Tkinter avec des onglets pour chaque module
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os

# Ajouter le répertoire parent au path pour importer les modules core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.linear_system import solve_linear_system
from core.linear_programming import LinearProgrammingSolver
from core.regression import LinearRegression, load_data_from_csv
from core.stochastic import MarkovChain, RandomWalk


class MathematicalApp:
    """
    Application principale avec interface graphique
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Application de Modélisation Mathématique")
        self.root.geometry("1200x800")
        
        # Créer le notebook (onglets)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Créer les onglets
        self.create_linear_system_tab()
        self.create_linear_programming_tab()
        self.create_regression_tab()
        self.create_stochastic_tab()
        
        # Barre de statut
        self.status_bar = tk.Label(self.root, text="Prêt", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_linear_system_tab(self):
        """
        Onglet pour la résolution de systèmes linéaires
        """
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Système Linéaire")
        
        # Frame gauche: Saisie
        left_frame = ttk.LabelFrame(tab, text="Saisie des données", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(left_frame, text="Taille de la matrice:").pack()
        self.ls_size = ttk.Spinbox(left_frame, from_=2, to=10, width=10)
        self.ls_size.set(3)
        self.ls_size.pack()
        
        ttk.Button(left_frame, text="Générer la grille", 
                  command=self.generate_linear_system_grid).pack(pady=5)
        
        # Frame pour la matrice
        self.ls_matrix_frame = ttk.Frame(left_frame)
        self.ls_matrix_frame.pack(pady=10)
        
        # Méthode de résolution
        ttk.Label(left_frame, text="Méthode:").pack()
        self.ls_method = ttk.Combobox(left_frame, values=['gauss', 'lu', 'numpy'], 
                                      state='readonly', width=15)
        self.ls_method.set('gauss')
        self.ls_method.pack()
        
        ttk.Button(left_frame, text="Résoudre", 
                  command=self.solve_linear_system, 
                  style='Accent.TButton').pack(pady=10)
        
        # Frame droite: Résultats
        right_frame = ttk.LabelFrame(tab, text="Résultats", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ls_result = scrolledtext.ScrolledText(right_frame, height=30, width=50)
        self.ls_result.pack(fill=tk.BOTH, expand=True)
        
        # Initialiser la grille
        self.ls_entries = []
        self.generate_linear_system_grid()
    
    def generate_linear_system_grid(self):
        """
        Génère la grille de saisie pour le système linéaire
        """
        # Nettoyer l'ancienne grille
        for widget in self.ls_matrix_frame.winfo_children():
            widget.destroy()
        
        n = int(self.ls_size.get())
        self.ls_entries = []
        
        # Créer les entrées pour la matrice A
        ttk.Label(self.ls_matrix_frame, text="Matrice A:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=n, pady=5)
        
        for i in range(n):
            row_entries = []
            for j in range(n):
                entry = ttk.Entry(self.ls_matrix_frame, width=8)
                entry.grid(row=i+1, column=j, padx=2, pady=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.ls_entries.append(row_entries)
        
        # Créer les entrées pour le vecteur b
        ttk.Label(self.ls_matrix_frame, text="=", font=('Arial', 12)).grid(
            row=1, column=n, padx=10)
        
        ttk.Label(self.ls_matrix_frame, text="Vecteur b:", font=('Arial', 10, 'bold')).grid(
            row=0, column=n+1, pady=5)
        
        self.ls_b_entries = []
        for i in range(n):
            entry = ttk.Entry(self.ls_matrix_frame, width=8)
            entry.grid(row=i+1, column=n+1, padx=5, pady=2)
            entry.insert(0, "0")
            self.ls_b_entries.append(entry)
        
        # Bouton pour exemple
        ttk.Button(self.ls_matrix_frame, text="Charger exemple", 
                  command=self.load_linear_system_example).grid(
            row=n+2, column=0, columnspan=n+2, pady=10)
    
    def load_linear_system_example(self):
        """
        Charge un exemple de système linéaire
        """
        n = int(self.ls_size.get())
        
        if n == 3:
            # Exemple 3x3
            A = [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]
            b = [1, -2, 0]
        elif n == 2:
            A = [[2, 1], [1, 3]]
            b = [5, 6]
        else:
            # Générer un exemple aléatoire
            A = np.random.randint(-5, 6, (n, n)).tolist()
            b = np.random.randint(-10, 11, n).tolist()
        
        for i in range(min(n, len(A))):
            for j in range(min(n, len(A[0]))):
                self.ls_entries[i][j].delete(0, tk.END)
                self.ls_entries[i][j].insert(0, str(A[i][j]))
        
        for i in range(min(n, len(b))):
            self.ls_b_entries[i].delete(0, tk.END)
            self.ls_b_entries[i].insert(0, str(b[i]))
    
    def solve_linear_system(self):
        """
        Résout le système linéaire
        """
        try:
            n = int(self.ls_size.get())
            
            # Lire la matrice A
            A = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    A[i, j] = float(self.ls_entries[i][j].get())
            
            # Lire le vecteur b
            b = np.zeros(n)
            for i in range(n):
                b[i] = float(self.ls_b_entries[i].get())
            
            # Résoudre
            method = self.ls_method.get()
            result = solve_linear_system(A, b, method)
            
            # Afficher les résultats
            self.ls_result.delete(1.0, tk.END)
            
            if result['success']:
                self.ls_result.insert(tk.END, f"=== RÉSOLUTION SYSTÈME LINÉAIRE ===\n\n")
                self.ls_result.insert(tk.END, f"Méthode utilisée: {result['method']}\n\n")
                
                self.ls_result.insert(tk.END, f"Matrice A:\n{A}\n\n")
                self.ls_result.insert(tk.END, f"Vecteur b: {b}\n\n")
                
                self.ls_result.insert(tk.END, "=" * 40 + "\n")
                self.ls_result.insert(tk.END, "SOLUTION:\n")
                for i, val in enumerate(result['solution']):
                    self.ls_result.insert(tk.END, f"  x{i+1} = {val:.6f}\n")
                
                self.ls_result.insert(tk.END, "\n" + "=" * 40 + "\n")
                self.ls_result.insert(tk.END, "INFORMATIONS:\n")
                self.ls_result.insert(tk.END, f"  Résidu: {result['residual']:.2e}\n")
                self.ls_result.insert(tk.END, f"  Déterminant: {result['determinant']:.6f}\n")
                self.ls_result.insert(tk.END, f"  Nombre de condition: {result['condition_number']:.2e}\n")
                
                self.status_bar.config(text="Système résolu avec succès")
            else:
                self.ls_result.insert(tk.END, f"ERREUR:\n{result['error']}")
                self.status_bar.config(text="Erreur lors de la résolution")
        
        except ValueError as e:
            messagebox.showerror("Erreur", f"Valeur invalide: {e}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur inattendue: {e}")
    
    def create_linear_programming_tab(self):
        """
        Onglet pour la programmation linéaire
        """
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Programmation Linéaire")
        
        # Frame gauche: Saisie
        left_frame = ttk.LabelFrame(tab, text="Définition du problème", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Fonction objectif
        ttk.Label(left_frame, text="Fonction objectif:", font=('Arial', 10, 'bold')).pack()
        
        obj_frame = ttk.Frame(left_frame)
        obj_frame.pack(pady=5)
        
        self.lp_sense = ttk.Combobox(obj_frame, values=['Maximiser', 'Minimiser'], 
                                     state='readonly', width=12)
        self.lp_sense.set('Maximiser')
        self.lp_sense.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(obj_frame, text="Z =").pack(side=tk.LEFT)
        self.lp_c1 = ttk.Entry(obj_frame, width=8)
        self.lp_c1.insert(0, "3")
        self.lp_c1.pack(side=tk.LEFT, padx=2)
        ttk.Label(obj_frame, text="x +").pack(side=tk.LEFT)
        self.lp_c2 = ttk.Entry(obj_frame, width=8)
        self.lp_c2.insert(0, "2")
        self.lp_c2.pack(side=tk.LEFT, padx=2)
        ttk.Label(obj_frame, text="y").pack(side=tk.LEFT)
        
        # Contraintes
        ttk.Label(left_frame, text="\nContraintes:", font=('Arial', 10, 'bold')).pack()
        
        self.lp_constraints_text = scrolledtext.ScrolledText(left_frame, height=10, width=40)
        self.lp_constraints_text.pack(pady=5, fill=tk.BOTH, expand=True)
        self.lp_constraints_text.insert(tk.END, "# Format: a*x + b*y <= c\n")
        self.lp_constraints_text.insert(tk.END, "2*x + 1*y <= 18\n")
        self.lp_constraints_text.insert(tk.END, "2*x + 3*y <= 42\n")
        self.lp_constraints_text.insert(tk.END, "3*x + 1*y <= 24\n")
        
        ttk.Button(left_frame, text="Résoudre", 
                  command=self.solve_linear_programming,
                  style='Accent.TButton').pack(pady=10)
        
        # Frame droite: Résultats
        right_frame = ttk.LabelFrame(tab, text="Résultats", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.lp_result = scrolledtext.ScrolledText(right_frame, height=30, width=50)
        self.lp_result.pack(fill=tk.BOTH, expand=True)
    
    def solve_linear_programming(self):
        """
        Résout le problème de programmation linéaire
        """
        try:
            # Lire la fonction objectif
            c1 = float(self.lp_c1.get())
            c2 = float(self.lp_c2.get())
            sense = 'maximize' if self.lp_sense.get() == 'Maximiser' else 'minimize'
            
            # Parser les contraintes
            constraints_text = self.lp_constraints_text.get(1.0, tk.END)
            constraints = []
            
            for line in constraints_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parser format: a*x + b*y <= c
                if '<=' in line:
                    left, right = line.split('<=')
                    comparator = '<='
                elif '>=' in line:
                    left, right = line.split('>=')
                    comparator = '>='
                elif '==' in line:
                    left, right = line.split('==')
                    comparator = '=='
                else:
                    continue
                
                # Extraire coefficients
                left = left.replace(' ', '')
                a = 0
                b = 0
                
                terms = left.replace('-', '+-').split('+')
                for term in terms:
                    if 'x' in term:
                        coef = term.replace('*x', '').replace('x', '')
                        a = float(coef) if coef and coef != '' else 1.0
                    elif 'y' in term:
                        coef = term.replace('*y', '').replace('y', '')
                        b = float(coef) if coef and coef != '' else 1.0
                
                c = float(right.strip())
                constraints.append(({'x': a, 'y': b}, comparator, c))
            
            # Résoudre
            solver = LinearProgrammingSolver()
            objective = {'x': c1, 'y': c2}
            result = solver.solve_standard_problem(objective, constraints, sense)
            
            # Afficher les résultats
            self.lp_result.delete(1.0, tk.END)
            
            if result['success']:
                self.lp_result.insert(tk.END, "=== PROGRAMMATION LINÉAIRE ===\n\n")
                self.lp_result.insert(tk.END, f"{self.lp_sense.get()} Z = {c1}x + {c2}y\n\n")
                
                self.lp_result.insert(tk.END, "Contraintes:\n")
                for coeffs, comp, val in constraints:
                    self.lp_result.insert(tk.END, 
                        f"  {coeffs['x']}x + {coeffs['y']}y {comp} {val}\n")
                
                self.lp_result.insert(tk.END, "\n" + "=" * 40 + "\n")
                self.lp_result.insert(tk.END, f"Statut: {result['status']}\n\n")
                self.lp_result.insert(tk.END, f"Valeur optimale: Z = {result['objective_value']:.4f}\n\n")
                
                self.lp_result.insert(tk.END, "Solution optimale:\n")
                for var, value in result['solution'].items():
                    self.lp_result.insert(tk.END, f"  {var} = {value:.4f}\n")
                
                self.status_bar.config(text="Problème résolu avec succès")
            else:
                self.lp_result.insert(tk.END, f"Statut: {result['status']}\n")
                self.lp_result.insert(tk.END, f"\n{result.get('message', result.get('error', ''))}")
                self.status_bar.config(text="Aucune solution trouvée")
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur: {e}")
    
    def create_regression_tab(self):
        """
        Onglet pour la régression linéaire
        """
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Régression Linéaire")
        
        # Frame haut: Contrôles
        top_frame = ttk.LabelFrame(tab, text="Données", padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(top_frame, text="Charger CSV", 
                  command=self.load_regression_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Générer données aléatoires", 
                  command=self.generate_regression_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Calculer régression", 
                  command=self.calculate_regression,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        
        # Frame milieu: Graphique
        middle_frame = ttk.LabelFrame(tab, text="Graphique", padding=10)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.reg_figure = plt.Figure(figsize=(8, 5))
        self.reg_canvas = FigureCanvasTkAgg(self.reg_figure, middle_frame)
        self.reg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame bas: Résultats
        bottom_frame = ttk.LabelFrame(tab, text="Statistiques", padding=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.reg_result = scrolledtext.ScrolledText(bottom_frame, height=8)
        self.reg_result.pack(fill=tk.BOTH, expand=True)
        
        # Variables pour stocker les données
        self.reg_X = None
        self.reg_y = None
        self.reg_model = None
    
    def load_regression_csv(self):
        """
        Charge des données depuis un fichier CSV
        """
        filepath = filedialog.askopenfilename(
            title="Sélectionner un fichier CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.reg_X, self.reg_y = load_data_from_csv(filepath)
                messagebox.showinfo("Succès", f"{len(self.reg_X)} points chargés")
                self.status_bar.config(text=f"Données chargées: {len(self.reg_X)} points")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le fichier: {e}")
    
    def generate_regression_data(self):
        """
        Génère des données aléatoires pour la régression
        """
        np.random.seed()
        self.reg_X = np.linspace(0, 10, 50)
        self.reg_y = 2.5 * self.reg_X + 1.0 + np.random.normal(0, 2, 50)
        messagebox.showinfo("Succès", "Données aléatoires générées (y = 2.5x + 1 + bruit)")
        self.status_bar.config(text="Données générées: 50 points")
    
    def calculate_regression(self):
        """
        Calcule et affiche la régression linéaire
        """
        if self.reg_X is None or self.reg_y is None:
            messagebox.showwarning("Attention", "Veuillez charger ou générer des données d'abord")
            return
        
        try:
            # Créer et ajuster le modèle
            self.reg_model = LinearRegression()
            self.reg_model.fit(self.reg_X, self.reg_y)
            
            # Tracer le graphique
            self.reg_figure.clear()
            ax = self.reg_figure.add_subplot(111)
            
            # Points de données
            ax.scatter(self.reg_X, self.reg_y, alpha=0.6, label='Données')
            
            # Droite de régression
            X_line = np.linspace(self.reg_X.min(), self.reg_X.max(), 100)
            y_line = self.reg_model.predict(X_line)
            ax.plot(X_line, y_line, 'r-', linewidth=2, label='Régression')
            
            ax.set_xlabel('X', fontsize=11)
            ax.set_ylabel('y', fontsize=11)
            ax.set_title(f'Régression Linéaire\n{self.reg_model.get_equation()}\nR² = {self.reg_model.r_squared:.4f}', 
                        fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.reg_canvas.draw()
            
            # Afficher les statistiques
            stats = self.reg_model.get_statistics()
            self.reg_result.delete(1.0, tk.END)
            self.reg_result.insert(tk.END, "=== STATISTIQUES DE RÉGRESSION ===\n\n")
            self.reg_result.insert(tk.END, f"Équation: {stats['equation']}\n\n")
            self.reg_result.insert(tk.END, f"Coefficient de détermination (R²): {stats['r_squared']:.6f}\n")
            self.reg_result.insert(tk.END, f"RMSE: {stats['rmse']:.6f}\n")
            self.reg_result.insert(tk.END, f"MAE: {stats['mae']:.6f}\n")
            self.reg_result.insert(tk.END, f"Nombre d'échantillons: {stats['n_samples']}\n")
            
            self.status_bar.config(text="Régression calculée avec succès")
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du calcul: {e}")
    
    def create_stochastic_tab(self):
        """
        Onglet pour les processus stochastiques
        """
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Processus Stochastique")
        
        # Sous-onglets
        sub_notebook = ttk.Notebook(tab)
        sub_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Sous-onglet Markov
        self.create_markov_subtab(sub_notebook)
        
        # Sous-onglet Marche aléatoire
        self.create_random_walk_subtab(sub_notebook)
    
    def create_markov_subtab(self, parent):
        """
        Sous-onglet pour les chaînes de Markov
        """
        tab = ttk.Frame(parent)
        parent.add(tab, text="Chaîne de Markov")
        
        # Frame gauche: Configuration
        left_frame = ttk.LabelFrame(tab, text="Configuration", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(left_frame, text="Nombre d'états:").pack()
        self.markov_states = ttk.Spinbox(left_frame, from_=2, to=5, width=10)
        self.markov_states.set(3)
        self.markov_states.pack()
        
        ttk.Button(left_frame, text="Générer matrice", 
                  command=self.generate_markov_matrix).pack(pady=5)
        
        self.markov_matrix_frame = ttk.Frame(left_frame)
        self.markov_matrix_frame.pack(pady=10)
        
        ttk.Label(left_frame, text="Nombre de transitions:").pack()
        self.markov_steps = ttk.Entry(left_frame, width=10)
        self.markov_steps.insert(0, "100")
        self.markov_steps.pack()
        
        ttk.Button(left_frame, text="Simuler", 
                  command=self.simulate_markov,
                  style='Accent.TButton').pack(pady=10)
        
        ttk.Button(left_frame, text="Charger exemple (météo)", 
                  command=self.load_markov_example).pack()
        
        # Frame droite: Résultats
        right_frame = ttk.Frame(tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Graphique
        graph_frame = ttk.LabelFrame(right_frame, text="Visualisation", padding=10)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        self.markov_figure = plt.Figure(figsize=(8, 6))
        self.markov_canvas = FigureCanvasTkAgg(self.markov_figure, graph_frame)
        self.markov_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistiques
        stats_frame = ttk.LabelFrame(right_frame, text="Statistiques", padding=10)
        stats_frame.pack(fill=tk.X)
        
        self.markov_result = scrolledtext.ScrolledText(stats_frame, height=8)
        self.markov_result.pack(fill=tk.BOTH, expand=True)
        
        # Variables
        self.markov_entries = []
        self.markov_chain = None
        
        # Initialiser
        self.generate_markov_matrix()
    
    def generate_markov_matrix(self):
        """
        Génère la grille pour la matrice de transition
        """
        for widget in self.markov_matrix_frame.winfo_children():
            widget.destroy()
        
        n = int(self.markov_states.get())
        self.markov_entries = []
        
        ttk.Label(self.markov_matrix_frame, text="Matrice de transition:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=n+1, pady=5)
        
        # En-têtes colonnes
        for j in range(n):
            ttk.Label(self.markov_matrix_frame, text=f"E{j}").grid(row=1, column=j+1)
        
        for i in range(n):
            # En-tête ligne
            ttk.Label(self.markov_matrix_frame, text=f"E{i}:").grid(row=i+2, column=0, padx=5)
            
            row_entries = []
            for j in range(n):
                entry = ttk.Entry(self.markov_matrix_frame, width=8)
                entry.grid(row=i+2, column=j+1, padx=2, pady=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.markov_entries.append(row_entries)
    
    def load_markov_example(self):
        """
        Charge un exemple de chaîne de Markov (météo)
        """
        # Matrice 3x3 (Soleil, Nuageux, Pluie)
        transition = [
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5]
        ]
        
        if int(self.markov_states.get()) != 3:
            self.markov_states.delete(0, tk.END)
            self.markov_states.insert(0, "3")
            self.generate_markov_matrix()
        
        for i in range(3):
            for j in range(3):
                self.markov_entries[i][j].delete(0, tk.END)
                self.markov_entries[i][j].insert(0, str(transition[i][j]))
        
        messagebox.showinfo("Exemple chargé", 
                          "Exemple météo:\nÉtat 0: Soleil\nÉtat 1: Nuageux\nÉtat 2: Pluie")
    
    def simulate_markov(self):
        """
        Simule la chaîne de Markov
        """
        try:
            n = int(self.markov_states.get())
            
            # Lire la matrice de transition
            transition = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    transition[i, j] = float(self.markov_entries[i][j].get())
            
            # Créer et simuler la chaîne
            steps = int(self.markov_steps.get())
            self.markov_chain = MarkovChain(transition)
            self.markov_chain.simulate(steps)
            
            # Tracer les graphiques
            self.markov_figure.clear()
            
            # Graphique 1: Trajectoire
            ax1 = self.markov_figure.add_subplot(2, 1, 1)
            history = self.markov_chain.history
            ax1.plot(history[:min(200, len(history))], 'o-', markersize=3, linewidth=0.8)
            ax1.set_xlabel('Temps')
            ax1.set_ylabel('État')
            ax1.set_title('Trajectoire (premiers pas)')
            ax1.set_yticks(range(n))
            ax1.grid(True, alpha=0.3)
            
            # Graphique 2: Distribution
            ax2 = self.markov_figure.add_subplot(2, 1, 2)
            state_counts = np.bincount(history, minlength=n)
            state_freq = state_counts / len(history)
            
            x_pos = np.arange(n)
            ax2.bar(x_pos, state_freq, alpha=0.7, label='Fréquence observée')
            
            # Distribution stationnaire
            try:
                stationary = self.markov_chain.get_stationary_distribution()
                ax2.plot(x_pos, stationary, 'ro-', linewidth=2, markersize=8,
                        label='Distribution stationnaire')
            except:
                pass
            
            ax2.set_xlabel('État')
            ax2.set_ylabel('Fréquence')
            ax2.set_title('Distribution des états')
            ax2.set_xticks(x_pos)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            self.markov_figure.tight_layout()
            self.markov_canvas.draw()
            
            # Afficher les statistiques
            stats = self.markov_chain.get_statistics()
            self.markov_result.delete(1.0, tk.END)
            self.markov_result.insert(tk.END, "=== SIMULATION CHAÎNE DE MARKOV ===\n\n")
            self.markov_result.insert(tk.END, f"Transitions: {stats['n_steps']}\n")
            self.markov_result.insert(tk.END, f"État initial: {stats['initial_state']}\n")
            self.markov_result.insert(tk.END, f"État final: {stats['final_state']}\n\n")
            
            self.markov_result.insert(tk.END, "Fréquences observées:\n")
            for i, freq in enumerate(stats['state_frequencies']):
                self.markov_result.insert(tk.END, f"  État {i}: {freq:.4f}\n")
            
            if stats['stationary_distribution'] is not None:
                self.markov_result.insert(tk.END, "\nDistribution stationnaire:\n")
                for i, prob in enumerate(stats['stationary_distribution']):
                    self.markov_result.insert(tk.END, f"  État {i}: {prob:.4f}\n")
            
            self.status_bar.config(text="Simulation Markov terminée")
        
        except ValueError as e:
            messagebox.showerror("Erreur", f"Matrice invalide: {e}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur: {e}")
    
    def create_random_walk_subtab(self, parent):
        """
        Sous-onglet pour les marches aléatoires
        """
        tab = ttk.Frame(parent)
        parent.add(tab, text="Marche Aléatoire")
        
        # Frame gauche: Configuration
        left_frame = ttk.LabelFrame(tab, text="Configuration", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(left_frame, text="Dimension:").pack()
        self.walk_dimension = ttk.Combobox(left_frame, values=['1D', '2D'], 
                                          state='readonly', width=10)
        self.walk_dimension.set('2D')
        self.walk_dimension.pack()
        
        ttk.Label(left_frame, text="\nNombre de pas:").pack()
        self.walk_steps = ttk.Entry(left_frame, width=10)
        self.walk_steps.insert(0, "500")
        self.walk_steps.pack()
        
        ttk.Label(left_frame, text="\nTaille du pas:").pack()
        self.walk_step_size = ttk.Entry(left_frame, width=10)
        self.walk_step_size.insert(0, "1.0")
        self.walk_step_size.pack()
        
        ttk.Button(left_frame, text="Simuler", 
                  command=self.simulate_random_walk,
                  style='Accent.TButton').pack(pady=20)
        
        # Frame droite: Visualisation
        right_frame = ttk.LabelFrame(tab, text="Visualisation", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.walk_figure = plt.Figure(figsize=(8, 6))
        self.walk_canvas = FigureCanvasTkAgg(self.walk_figure, right_frame)
        self.walk_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Variable
        self.random_walk = None
    
    def simulate_random_walk(self):
        """
        Simule une marche aléatoire
        """
        try:
            dimension = 1 if self.walk_dimension.get() == '1D' else 2
            steps = int(self.walk_steps.get())
            step_size = float(self.walk_step_size.get())
            
            # Créer et simuler
            self.random_walk = RandomWalk(dimension)
            self.random_walk.simulate(steps, step_size)
            
            # Tracer
            self.walk_figure.clear()
            ax = self.walk_figure.add_subplot(111)
            
            if dimension == 1:
                ax.plot(self.random_walk.trajectory, linewidth=1.5)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel('Temps')
                ax.set_ylabel('Position')
                ax.set_title('Marche Aléatoire 1D')
                ax.grid(True, alpha=0.3)
            
            else:  # 2D
                x = self.random_walk.trajectory[:, 0]
                y = self.random_walk.trajectory[:, 1]
                
                points = ax.scatter(x, y, c=range(len(x)), cmap='viridis', 
                                  s=10, alpha=0.6)
                ax.plot(x, y, 'b-', alpha=0.2, linewidth=0.5)
                
                ax.plot(x[0], y[0], 'go', markersize=12, label='Début', zorder=5)
                ax.plot(x[-1], y[-1], 'ro', markersize=12, label='Fin', zorder=5)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Marche Aléatoire 2D\nDistance finale: {np.linalg.norm([x[-1], y[-1]]):.2f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axis('equal')
                
                self.walk_figure.colorbar(points, label='Temps')
            
            self.walk_figure.tight_layout()
            self.walk_canvas.draw()
            
            self.status_bar.config(text="Marche aléatoire simulée")
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur: {e}")


def main():
    """
    Point d'entrée de l'application
    """
    root = tk.Tk()
    app = MathematicalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
