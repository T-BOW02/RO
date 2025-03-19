import numpy as np
import csv
import gurobipy as gp
from gurobipy import GRB


# Lecture des données d'instance
def read_instance(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(val) for val in row])
    return np.array(data)


# Résolution avec l'approche epsilon-contrainte pour problème bi-objectif
def solve_with_gurobi(instance_data, epsilon=None, max_makespan=None):
    n_jobs = len(instance_data)
    n_machines = 10  # Nombre de machines fixé à 10

    # Extraire les données
    priorities = instance_data[:, 0]
    deadlines = instance_data[:, 1]
    processing_times = instance_data[:, 2:2 + n_machines]

    # Calculer une borne supérieure pour le makespan si non fournie
    if max_makespan is None:
        max_makespan = sum(processing_times.sum(axis=1))

    # Créer le modèle
    model = gp.Model("FlowShop_BiObjective")

    # Variables de décision
    # x[i,j] = 1 si le job i est à la position j dans la séquence
    x = model.addVars(n_jobs, n_jobs, vtype=GRB.BINARY, name="x")

    # C[j,k] = temps d'achèvement du job en position j sur la machine k
    C = model.addVars(n_jobs, n_machines, lb=0, vtype=GRB.CONTINUOUS, name="C")

    # T[j] = retard du job en position j
    T = model.addVars(n_jobs, lb=0, vtype=GRB.CONTINUOUS, name="T")

    # Contraintes d'affectation
    # Chaque job doit être affecté à exactement une position
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n_jobs)) == 1
                      for i in range(n_jobs)), name="assign_job")

    # Chaque position doit contenir exactement un job
    model.addConstrs((gp.quicksum(x[i, j] for i in range(n_jobs)) == 1
                      for j in range(n_jobs)), name="fill_position")

    # Contraintes de flow shop
    # Temps d'achèvement sur la première machine
    model.addConstr(C[0, 0] >= gp.quicksum(x[i, 0] * processing_times[i, 0]
                                           for i in range(n_jobs)), name="start_machine_0")

    # Temps d'achèvement sur les autres machines pour le premier job
    for k in range(1, n_machines):
        model.addConstr(C[0, k] >= C[0, k - 1] + gp.quicksum(x[i, 0] * processing_times[i, k]
                                                             for i in range(n_jobs)),
                        name=f"first_job_machine_{k}")

    # Temps d'achèvement pour les jobs suivants sur la première machine
    for j in range(1, n_jobs):
        model.addConstr(C[j, 0] >= C[j - 1, 0] + gp.quicksum(x[i, j] * processing_times[i, 0]
                                                             for i in range(n_jobs)),
                        name=f"job_{j}_machine_0")

    # Temps d'achèvement pour les jobs et machines restants
    for j in range(1, n_jobs):
        for k in range(1, n_machines):
            model.addConstr(C[j, k] >= C[j - 1, k] + gp.quicksum(x[i, j] * processing_times[i, k]
                                                                 for i in range(n_jobs)),
                            name=f"job_{j}_machine_{k}_precedence1")
            model.addConstr(C[j, k] >= C[j, k - 1] + gp.quicksum(x[i, j] * processing_times[i, k]
                                                                 for i in range(n_jobs)),
                            name=f"job_{j}_machine_{k}_precedence2")

    # Calcul du retard
    for j in range(n_jobs):
        model.addConstr(T[j] >= C[j, n_machines - 1] - gp.quicksum(x[i, j] * deadlines[i]
                                                                   for i in range(n_jobs)),
                        name=f"tardiness_{j}")

    # Fonction objectif: approche epsilon-contrainte
    # Si epsilon est fourni, on minimise le makespan avec une contrainte sur la tardiveté
    # Sinon, on minimise la tardiveté pondérée avec une contrainte sur le makespan

    if epsilon is not None:
        # Contrainte sur la tardiveté pondérée totale
        model.addConstr(gp.quicksum(T[j] * gp.quicksum(x[i, j] * priorities[i]
                                                       for i in range(n_jobs))
                                    for j in range(n_jobs)) <= epsilon,
                        name="epsilon_constraint")
        # Minimiser le makespan
        model.setObjective(C[n_jobs - 1, n_machines - 1], GRB.MINIMIZE)
    else:
        # Contrainte sur le makespan
        if max_makespan:
            model.addConstr(C[n_jobs - 1, n_machines - 1] <= max_makespan,
                            name="makespan_constraint")
        # Minimiser la tardiveté pondérée totale
        model.setObjective(gp.quicksum(T[j] * gp.quicksum(x[i, j] * priorities[i]
                                                          for i in range(n_jobs))
                                       for j in range(n_jobs)), GRB.MINIMIZE)

    # Paramètres du solveur
    model.setParam('TimeLimit', 3600)  # Limite de temps à 1 heure

    # Optimisation
    model.optimize()

    # Récupération des résultats
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        sequence = [-1] * n_jobs
        for i in range(n_jobs):
            for j in range(n_jobs):
                if x[i, j].X > 0.5:  # Variable binaire activée
                    sequence[j] = i

        makespan = C[n_jobs - 1, n_machines - 1].X
        total_weighted_tardiness = model.getObjective().getValue() if epsilon is not None else sum(
            T[j].X * sum(x[i, j].X * priorities[i] for i in range(n_jobs)) for j in range(n_jobs))

        return sequence, makespan, total_weighted_tardiness
    else:
        print(f"Optimization failed with status {model.status}")
        return None, None, None


# Génération du front de Pareto en utilisant l'approche epsilon-contrainte
def generate_pareto_front_gurobi(instance_data, n_points=10):
    # Résoudre en minimisant la tardiveté pondérée sans contrainte sur le makespan
    _, _, min_tard = solve_with_gurobi(instance_data)

    # Résoudre en minimisant le makespan sans contrainte sur la tardiveté
    _, min_makespan, _ = solve_with_gurobi(instance_data, epsilon=float('inf'))

    # Générer les points intermédiaires du front de Pareto
    pareto_front = []
    epsilons = [min_tard + i * (float('inf') - min_tard) / (n_points - 1) for i in range(n_points)]

    for eps in epsilons:
        sequence, makespan, tardiness = solve_with_gurobi(instance_data, epsilon=eps)
        if sequence:
            pareto_front.append((sequence, makespan, tardiness))

    # Trier par makespan croissant
    pareto_front.sort(key=lambda x: x[1])

    return pareto_front


# Fonction principale
def main():
    # Lecture de l'instance
    instance_data = read_instance("instance.csv")

    # Génération du front de Pareto
    pareto_front = generate_pareto_front_gurobi(instance_data, n_points=5)

    # Affichage des résultats
    print("Solutions Pareto trouvées avec Gurobi:")
    for i, (sequence, makespan, tardiness) in enumerate(pareto_front):
        print(f"Solution {i + 1}:")
        print(f"  Séquence: {sequence}")
        print(f"  Makespan: {makespan}")
        print(f"  Tardiveté pondérée totale: {tardiness}")

    # Sauvegarde des résultats
    with open("Gurobi_Pareto_Front.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        for solution, _, _ in pareto_front:
            writer.writerow(solution)
    print("Front de Pareto sauvegardé dans 'Gurobi_Pareto_Front.csv'")


if __name__ == "__main__":
    main()
