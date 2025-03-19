
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize

import time

def charger_donnees(fichier_instance):
    try:
        data = pd.read_csv(fichier_instance)
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        # Création de données fictives pour la démonstration
        n_pieces = 200
        n_machines = 10
        priorites = np.random.randint(1, 10, size=n_pieces)
        deadlines = np.random.randint(100, 1000, size=n_pieces)
        temps_usinage = np.random.randint(1, 100, size=(n_pieces, n_machines))
        columns = ['priority', 'deadline'] + [f'machine_{i+1}' for i in range(n_machines)]
        data = pd.DataFrame(np.hstack((priorites.reshape(-1, 1),
                                     deadlines.reshape(-1, 1),
                                     temps_usinage)), columns=columns)
        return data


def calculer_objectifs(permutation, donnees):
    n_pieces = len(permutation)
    n_machines = 10

    # Extraire les priorités et deadlines
    priorites = donnees.iloc[:, 0].values
    deadlines = donnees.iloc[:, 1].values

    # Extraire les temps d'usinage
    temps_usinage = donnees.iloc[:, 2:12].values

    # Initialiser les temps de fin pour chaque machine
    temps_fin = np.zeros((n_pieces, n_machines))

    # Calculer les temps de fin pour la première pièce
    for j in range(n_machines):
        if j == 0:
            temps_fin[0, j] = temps_usinage[permutation[0], j]
        else:
            temps_fin[0, j] = temps_fin[0, j - 1] + temps_usinage[permutation[0], j]

    # Calculer les temps de fin pour les pièces suivantes
    for i in range(1, n_pieces):
        for j in range(n_machines):
            if j == 0:
                temps_fin[i, j] = temps_fin[i - 1, j] + temps_usinage[permutation[i], j]
            else:
                temps_fin[i, j] = max(temps_fin[i, j - 1], temps_fin[i - 1, j]) + \
                                  temps_usinage[permutation[i], j]

    # Makespan: temps de fin de la dernière pièce sur la dernière machine
    makespan = temps_fin[-1, -1]

    # Total weighted tardiness
    total_weighted_tardiness = 0
    for i in range(n_pieces):
        tardiness = max(0, temps_fin[i, -1] - deadlines[permutation[i]])
        total_weighted_tardiness += tardiness * priorites[permutation[i]]

    return makespan, total_weighted_tardiness


class FlowshopProblem(Problem):
    def __init__(self, donnees):
        super().__init__(n_var=len(donnees), n_obj=2, n_constr=0,
                         xl=0, xu=len(donnees) - 1, type_var=int)
        self.donnees = donnees

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.zeros((x.shape[0], 2))

        for i in range(x.shape[0]):
            # Vérifier si la permutation est valide
            if len(np.unique(x[i])) != len(self.donnees):
                # Pénalité pour les solutions invalides
                f[i, 0] = 1e10
                f[i, 1] = 1e10
                continue

            # Calculer les objectifs
            makespan, twt = calculer_objectifs(x[i].astype(int), self.donnees)
            f[i, 0] = makespan
            f[i, 1] = twt

        out["F"] = f


def resoudre_nsga2(donnees, pop_size=100, n_gen=100):
    problem = FlowshopProblem(donnees)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True
    )

    start_time = time.time()
    res = minimize(problem, algorithm, ('n_gen', n_gen), verbose=True)
    end_time = time.time()

    print(f"Temps d'exécution NSGA-II: {end_time - start_time:.2f} secondes")

    # Extraire les solutions Pareto optimales
    X = res.X
    F = res.F

    return X, F


def charger_solution_utilisateur(fichier_solution, donnees):
    try:
        solutions = pd.read_csv(fichier_solution, header=None)

        X = []
        F = []

        for i in range(len(solutions)):
            permutation = solutions.iloc[i].values
            makespan, twt = calculer_objectifs(permutation, donnees)

            X.append(permutation)
            F.append([makespan, twt])

        return np.array(X), np.array(F)
    except Exception as e:
        print(f"Erreur lors du chargement de la solution utilisateur: {e}")
        print("Génération d'une solution aléatoire pour la démonstration...")

        # Créer une solution aléatoire
        n_pieces = len(donnees)
        n_solutions = 5

        X = []
        F = []

        for _ in range(n_solutions):
            perm = list(range(n_pieces))
            np.random.shuffle(perm)

            makespan, twt = calculer_objectifs(perm, donnees)

            X.append(perm)
            F.append([makespan, twt])

        return np.array(X), np.array(F)


def visualiser_resultats(resultats, labels):
    plt.figure(figsize=(10, 6))

    for i, (F, label) in enumerate(zip(resultats, labels)):
        plt.scatter(F[:, 0], F[:, 1], label=f"{label} ({len(F)} solutions)")

    plt.xlabel('Makespan')
    plt.ylabel('Total Weighted Tardiness')
    plt.title('Comparaison des Frontières Pareto')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparaison_solutions.png')
    plt.show()


def main():
    print("Résolution du problème de Flowshop Bi-objectif")
    print("==============================================\n")

    # Charger les données
    print("Chargement des données...")
    donnees = charger_donnees("instance.csv")
    print(f"Données chargées: {len(donnees)} pièces\n")

    # Paramètres
    pop_size = 50  # Taille de la population
    n_gen = 30  # Nombre de générations

    # Résoudre avec NSGA-II
    print("Résolution avec NSGA-II...")
    X_nsga2, F_nsga2 = resoudre_nsga2(donnees, pop_size, n_gen)

    # Charger la solution de l'utilisateur
    print("\nChargement de votre solution...")
    X_user, F_user = charger_solution_utilisateur("ma_solution.csv", donnees)

    # Visualiser les résultats
    print("\nVisualisation des résultats...")
    visualiser_resultats([F_nsga2, F_user], ["NSGA-II", "Votre solution"])

    # Afficher quelques statistiques
    print("\nStatistiques NSGA-II:")
    print(f"Nombre de solutions: {len(F_nsga2)}")
    print(f"Makespan min: {np.min(F_nsga2[:, 0]):.2f}")
    print(f"Makespan max: {np.max(F_nsga2[:, 0]):.2f}")
    print(f"TWT min: {np.min(F_nsga2[:, 1]):.2f}")
    print(f"TWT max: {np.max(F_nsga2[:, 1]):.2f}")

    print("\nStatistiques de votre solution:")
    print(f"Nombre de solutions: {len(F_user)}")
    print(f"Makespan min: {np.min(F_user[:, 0]):.2f}")
    print(f"Makespan max: {np.max(F_user[:, 0]):.2f}")
    print(f"TWT min: {np.min(F_user[:, 1]):.2f}")
    print(f"TWT max: {np.max(F_user[:, 1]):.2f}")

    # Sauvegarder les solutions NSGA-II
    print("\nSauvegarde des solutions NSGA-II...")
    with open("solutions_nsga2.csv", "w") as f:
        for sol in X_nsga2:
            f.write(",".join(map(str, sol)) + "\n")

    print("Solutions sauvegardées dans 'solutions_nsga2.csv'")
    print("\nTerminé!")


if __name__ == "__main__":
    main()
