"""
Module: tlbo_optimizer.py
Description: Teaching-Learning-Based Optimization (TLBO) for selecting the optimal 
5 acoustic features out of 29 by minimizing the intra-class to inter-class scatter ratio.
Includes iteration-by-iteration convergence history tracking.
"""

import numpy as np


# ---------------------------------------------------------
# Section 1: Objective Fitness Function (Scatter Ratio)
# ---------------------------------------------------------

def compute_scatter_ratio(X_subset: np.ndarray, y: np.ndarray) -> float:
    """
    Computes fitness objective: J = S_w / S_b (Intra-class Scatter / Inter-class Scatter).
    Minimizing J maximizes separation between Healthy and PD while minimizing within-class spread.
    """
    if X_subset.shape[1] == 0:
        return 1e6

    classes = np.unique(y)
    overall_mean = np.mean(X_subset, axis=0)

    s_w = 0.0  # Intra-class scatter (within class)
    s_b = 0.0  # Inter-class scatter (between class)

    for c in classes:
        X_c = X_subset[y == c]
        n_c = X_c.shape[0]
        if n_c == 0:
            continue
        mean_c = np.mean(X_c, axis=0)
        
        # Intra-class variance (sum of squared deviations from class centroid)
        s_w += np.sum((X_c - mean_c) ** 2)
        # Inter-class variance (weighted distance of class centroid to global centroid)
        s_b += n_c * np.sum((mean_c - overall_mean) ** 2)

    if s_b < 1e-8:
        return 1e6

    return float(s_w / s_b)


# ---------------------------------------------------------
# Section 2: TLBO Algorithm Implementation
# ---------------------------------------------------------

class TLBOFeatureSelector:
    """
    Continuous TLBO feature selection mapped to top-K feature indices.
    """
    def __init__(self, n_learners: int = 50, max_iter: int = 200, target_features: int = 5, random_state: int = 42):
        self.n_learners = n_learners
        self.max_iter = max_iter
        self.target_features = target_features
        self.random_state = random_state
        self.best_indices = None
        self.best_fitness = float("inf")
        self.history = []

    def _get_top_indices(self, continuous_vector: np.ndarray) -> np.ndarray:
        """
        Maps continuous learner position vector [0, 1]^D to the top-K feature indices.
        """
        return np.argsort(continuous_vector)[-self.target_features:]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Executes Teacher Phase and Learner Phase optimization iterations
        and logs the minimum scatter ratio at each step.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Initialize learner population randomly in range [0, 1]
        population = np.random.rand(self.n_learners, n_features)
        fitness = np.zeros(self.n_learners)

        # Initial fitness evaluation
        for i in range(self.n_learners):
            selected_idx = self._get_top_indices(population[i])
            fitness[i] = compute_scatter_ratio(X[:, selected_idx], y)

        self.history = []

        for iteration in range(self.max_iter):
            # Identify current Teacher (learner with lowest scatter ratio)
            best_idx = np.argmin(fitness)
            teacher = population[best_idx].copy()
            self.history.append(float(fitness[best_idx]))

            # ----------------- Phase 1: Teacher Phase -----------------
            class_mean = np.mean(population, axis=0)
            for i in range(self.n_learners):
                t_f = np.random.choice([1, 2])  # Teaching factor
                r = np.random.rand(n_features)
                step = r * (teacher - (t_f * class_mean))
                new_learner = np.clip(population[i] + step, 0.0, 1.0)

                new_idx = self._get_top_indices(new_learner)
                new_fit = compute_scatter_ratio(X[:, new_idx], y)

                if new_fit < fitness[i]:
                    population[i] = new_learner
                    fitness[i] = new_fit

            # ----------------- Phase 2: Learner Phase -----------------
            for i in range(self.n_learners):
                peer_idx = np.random.choice([idx for idx in range(self.n_learners) if idx != i])
                r = np.random.rand(n_features)

                # Move towards better peer or away from worse peer
                if fitness[i] < fitness[peer_idx]:
                    step = r * (population[i] - population[peer_idx])
                else:
                    step = r * (population[peer_idx] - population[i])

                new_learner = np.clip(population[i] + step, 0.0, 1.0)
                new_idx = self._get_top_indices(new_learner)
                new_fit = compute_scatter_ratio(X[:, new_idx], y)

                if new_fit < fitness[i]:
                    population[i] = new_learner
                    fitness[i] = new_fit

        # Extract optimal feature subset found
        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_indices = self._get_top_indices(population[best_idx])
        self.history.append(self.best_fitness)
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduces input matrix from (N x 29) to (N x 5) using selected optimal indices.
        """
        if self.best_indices is None:
            raise ValueError("Optimizer has not been fitted yet.")
        return X[:, self.best_indices]