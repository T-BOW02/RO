import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import warnings

warnings.filterwarnings("ignore")


# Fonction pour lire l'instance
def read_instance(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(val) for val in row])
    return np.array(data)


# Fonction pour lire les solutions Pareto
def read_pareto_solutions(filename):
    solutions = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            solutions.append([int(val) for val in row])
    return solutions


# Fonction d'évaluation identique à celle de notre algorithme custom
def evaluate(permutation, instance_data):
    n_jobs = len(permutation)
    n_machines = 10

    completion_times = np.zeros((n_jobs, n_machines))

    first_job = permutation[0]
    for j in range(n_machines):
        if j == 0:
            completion_times[0, j] = instance_data[first_job, j + 2]
        else:
            completion_times[0, j] = completion_times[0, j - 1] + instance_data[first_job, j + 2]

    for i in range(1, n_jobs):
        job = permutation[i]
        for j in range(n_machines):
            if j == 0:
                completion_times[i, j] = completion_times[i - 1, j] + instance_data[job, j + 2]
            else:
                completion_times[i, j] = max(completion_times[i, j - 1], completion_times[i - 1, j]) + instance_data[
                    job, j + 2]

    makespan = completion_times[-1, -1]

    total_weighted_tardiness = 0
    for i in range(n_jobs):
        job = permutation[i]
        completion_time = completion_times[i, -1]
        priority = instance_data[job, 0]
        deadline = instance_data[job, 1]
        tardiness = max(0, completion_time - deadline)
        total_weighted_tardiness += tardiness * priority

    return makespan, total_weighted_tardiness


# Fonction pour tracer la comparaison des frontières Pareto
def plot_pareto_comparison(pareto_fronts, labels, filename="pareto_comparison.png"):
    plt.figure(figsize=(12, 8))

    markers = ['o', 's', '^', 'D', 'x', '+']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

    for i, (front, label) in enumerate(zip(pareto_fronts, labels)):
        makespan_values = [obj[0] for obj in front]
        tardiness_values = [obj[1] for obj in front]
        plt.scatter(makespan_values, tardiness_values, marker=markers[i], color=colors[i], label=label, alpha=0.7)
        plt.plot(makespan_values, tardiness_values, '--', color=colors[i], alpha=0.3)

    plt.xlabel('Makespan')
    plt.ylabel('Total Weighted Tardiness')
    plt.title('Comparaison des frontières Pareto')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.show()


# Chargement de l'instance et de la solution personnalisée
print("Chargement des données...")
instance_data = read_instance("instance.csv")
custom_solutions = read_pareto_solutions("nom1_prenom1_nom2_prenom2.csv")  # Remplacer par votre fichier

# Évaluation de la solution personnalisée
custom_objectives = []
for solution in custom_solutions:
    makespan, tardiness = evaluate(solution, instance_data)
    custom_objectives.append((makespan, tardiness))

# Liste pour stocker toutes les frontières Pareto
all_pareto_fronts = [custom_objectives]
all_labels = ["Algorithme Custom"]

print("Optimisation avec 5 bibliothèques différentes...")

# 1. DEAP
print("\n1. Optimisation avec DEAP...")
try:
    import random
    from deap import base, creator, tools, algorithms

    # Création des classes fitness et individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()


    # Génération des individus (permutations)
    def permutation_gen():
        return random.sample(range(len(instance_data)), len(instance_data))


    toolbox.register("indices", permutation_gen)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # Fonction d'évaluation
    def evalFlowshop(individual):
        return evaluate(individual, instance_data)


    toolbox.register("evaluate", evalFlowshop)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    # Exécution de l'algorithme
    start_time = time.time()

    pop = toolbox.population(n=100)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                        stats=stats, halloffame=hof, verbose=False)

    deap_time = time.time() - start_time
    print(f"Temps d'exécution: {deap_time:.2f} s")

    # Récupération des objectifs
    deap_objectives = []
    for ind in hof:
        makespan, tardiness = ind.fitness.values
        deap_objectives.append((abs(makespan), abs(tardiness)))  # Négation car DEAP minimise

    all_pareto_fronts.append(deap_objectives)
    all_labels.append("DEAP (NSGA-II)")

except Exception as e:
    print(f"Erreur avec DEAP: {e}")

# 2. Platypus
print("\n2. Optimisation avec Platypus...")
try:
    from platypus import NSGAII, Problem, Integer, nondominated


    # Définition du problème
    class FlowshopProblem(Problem):
        def __init__(self):
            super().__init__(len(instance_data), 2)
            self.types[:] = Integer(0, len(instance_data) - 1)
            self.directions[:] = Problem.MINIMIZE

        def evaluate(self, solution):
            # Vérifier que la solution est une permutation valide
            if len(set(solution.variables)) != len(solution.variables):
                solution.variables = random.sample(range(len(instance_data)), len(instance_data))

            makespan, tardiness = evaluate(solution.variables, instance_data)
            solution.objectives[:] = [makespan, tardiness]


    start_time = time.time()

    # Résolution
    problem = FlowshopProblem()
    algorithm = NSGAII(problem, population_size=100)
    algorithm.run(10000)

    platypus_time = time.time() - start_time
    print(f"Temps d'exécution: {platypus_time:.2f} s")

    # Récupération des solutions non dominées
    nondominated_solutions = nondominated(algorithm.result)

    platypus_objectives = []
    for solution in nondominated_solutions:
        platypus_objectives.append((solution.objectives[0], solution.objectives[1]))

    all_pareto_fronts.append(platypus_objectives)
    all_labels.append("Platypus (NSGA-II)")

except Exception as e:
    print(f"Erreur avec Platypus: {e}")

# 3. PyMOO
print("\n3. Optimisation avec PyMOO...")
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.pm import PolynomialMutation
    from pymoo.optimize import minimize


    # Définition du problème
    class FlowshopProblemPyMOO(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(instance_data),
                             n_obj=2,
                             n_constr=0,
                             xl=0,
                             xu=len(instance_data) - 1)

        def _evaluate(self, x, out, *args, **kwargs):
            # Vérification de la permutation
            if len(set(x)) != len(x):
                x = np.random.permutation(len(instance_data))

            makespan, tardiness = evaluate(x, instance_data)
            out["F"] = [makespan, tardiness]


    start_time = time.time()

    # Résolution
    problem = FlowshopProblemPyMOO()

    algorithm = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=PolynomialMutation(prob=1.0 / len(instance_data)),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 50),
                   seed=1,
                   verbose=False)

    pymoo_time = time.time() - start_time
    print(f"Temps d'exécution: {pymoo_time:.2f} s")

    # Récupération des objectifs
    pymoo_objectives = []
    for solution in res.F:
        pymoo_objectives.append((solution[0], solution[1]))

    all_pareto_fronts.append(pymoo_objectives)
    all_labels.append("PyMOO (NSGA-II)")

except Exception as e:
    print(f"Erreur avec PyMOO: {e}")

# 4. jMetalPy
print("\n4. Optimisation avec jMetalPy...")

try:
    import time
    import random
    from jmetal.algorithm.multiobjective.nsgaii import NSGAII as JMetalNSGAII
    from jmetal.operator import IntegerPolynomialMutation, PMXCrossover
    from jmetal.util.termination_criterion import StoppingByEvaluations
    from jmetal.core.problem import PermutationProblem
    from jmetal.core.solution import PermutationSolution
    import random
    # Exemple de données fictives (instance_data)
    instance_data = [
        [2, 4, 6],  # Exemple de temps de traitement pour chaque machine
        [3, 5, 7]
    ]

    # Fonction d'évaluation fictive (à adapter selon votre problème)
    def evaluate(solution_variables, instance_data):
        makespan = sum(solution_variables)  # Exemple : somme des variables comme makespan
        tardiness = sum([abs(x - i) for i, x in enumerate(solution_variables)])  # Exemple : retard pondéré
        return makespan, tardiness

    # Définition du problème
    class FlowshopProblemJMetal(PermutationProblem):
        def __init__(self):
            super(FlowshopProblemJMetal, self).__init__()
            self.number_of_variables = len(instance_data[0])  # Nombre de tâches
            self.number_of_objectives = 2  # Deux objectifs : Makespan et Tardiness
            self.number_of_constraints = 0  # Pas de contraintes dans ce cas
            self.obj_directions = [self.MINIMIZE, self.MINIMIZE]  # Minimiser les deux objectifs
            self.obj_labels = ['Makespan', 'Total Weighted Tardiness']

        def evaluate(self, solution):
            makespan, tardiness = evaluate(solution.variables, instance_data)
            solution.objectives[0] = makespan
            solution.objectives[1] = tardiness

        def create_solution(self):
            new_solution = PermutationSolution(
                number_of_variables=self.number_of_variables,
                number_of_objectives=self.number_of_objectives
            )
            new_solution.variables = random.sample(range(self.number_of_variables), self.number_of_variables)
            return new_solution

        def get_name(self) -> str:
            return "Flowshop Problem JMetal"

    start_time = time.time()

    # Résolution
    problem = FlowshopProblemJMetal()

    algorithm = JMetalNSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables),
        crossover=PMXCrossover(probability=0.9),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000)
    )

    algorithm.run()

    jmetal_time = time.time() - start_time
    print(f"Temps d'exécution: {jmetal_time:.2f} s")

    # Récupération des objectifs
    jmetal_objectives = []
    for solution in algorithm.get_result():
        jmetal_objectives.append((solution.objectives[0], solution.objectives[1]))

    all_pareto_fronts.append(jmetal_objectives)
    all_labels.append("jMetalPy (NSGA-II)")

except Exception as e:
    print(f"Erreur avec jMetalPy: {e}")


# 5. Gurobi (pour optimisation exacte)
print("\n5. Optimisation avec Gurobi...")
try:
    import gurobipy as gp
    from gurobipy import GRB

    # Approche simplifiée avec epsilon-constraint
    start_time = time.time()

    # Récupération de quelques points représentatifs de la frontière
    gurobi_objectives = []

    # Bornes approximatives pour le makespan
    min_makespan = 5000  # Valeur approximative pour l'exemple
    max_makespan = 15000  # Valeur approximative pour l'exemple

    # Génération de quelques points de la frontière Pareto
    n_points = 5  # Nombre limité à cause du temps de calcul

    for i in range(n_points):
        # Limite de makespan pour cette itération
        makespan_limit = min_makespan + (max_makespan - min_makespan) * i / (n_points - 1)

        # Création du modèle
        model = gp.Model("flowshop_epsilon_constraint")
        model.setParam('OutputFlag', 0)  # Désactiver les logs

        # Variables: permutation des pièces
        perm = {}
        for j in range(len(instance_data)):
            for p in range(len(instance_data)):
                perm[j, p] = model.addVar(vtype=GRB.BINARY, name=f"perm_{j}_{p}")

        # Contraintes: chaque position a exactement une pièce
        for p in range(len(instance_data)):
            model.addConstr(gp.quicksum(perm[j, p] for j in range(len(instance_data))) == 1)

        # Chaque pièce est utilisée exactement une fois
        for j in range(len(instance_data)):
            model.addConstr(gp.quicksum(perm[j, p] for p in range(len(instance_data))) == 1)

        # Objectif: minimiser tardiness (approximation)
        # Note: ceci est une approximation très simplifiée
        tardiness_obj = 0
        for j in range(len(instance_data)):
            priority = instance_data[j, 0]
            deadline = instance_data[j, 1]
            for p in range(len(instance_data)):
                # Temps estimé pour terminer à la position p
                estimated_finish = p * 1000  # Approximation grossière
                tardiness = max(0, estimated_finish - deadline)
                tardiness_obj += perm[j, p] * tardiness * priority

        model.setObjective(tardiness_obj, GRB.MINIMIZE)

        # Limite sur le makespan (approche epsilon-constraint)
        makespan_expr = 0
        for j in range(len(instance_data)):
            for p in range(len(instance_data)):
                # Contribution estimée au makespan
                makespan_expr += perm[j, p] * p * 100  # Approximation

        model.addConstr(makespan_expr <= makespan_limit)

        # Optimisation avec un timeout court
        model.setParam('TimeLimit', 30)  # 30 secondes max par point
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Reconstruction de la solution
            solution = [0] * len(instance_data)
            for j in range(len(instance_data)):
                for p in range(len(instance_data)):
                    if perm[j, p].X > 0.5:
                        solution[p] = j

            # Évaluation réelle de la solution
            real_makespan, real_tardiness = evaluate(solution, instance_data)
            gurobi_objectives.append((real_makespan, real_tardiness))

    gurobi_time = time.time() - start_time
    print(f"Temps d'exécution: {gurobi_time:.2f} s")

    all_pareto_fronts.append(gurobi_objectives)
    all_labels.append("Gurobi (ε-constraint)")

except Exception as e:
    print(f"Erreur avec Gurobi: {e}")

# Tracer la comparaison des frontières Pareto
print("\nComparaison des frontières Pareto...")
plot_pareto_comparison(all_pareto_fronts, all_labels)

# Résumé des résultats
print("\nRésumé des résultats:")
print("--------------------")
for i, (front, label) in enumerate(zip(all_pareto_fronts, all_labels)):
    print(f"{label}: {len(front)} solutions")
    if front:
        min_makespan = min(obj[0] for obj in front)
        min_tardiness = min(obj[1] for obj in front)
        print(f"  - Makespan min: {min_makespan:.2f}")
        print(f"  - Total Weighted Tardiness min: {min_tardiness:.2f}")
    print()

print("Note: Ce script est uniquement à des fins de comparaison et d'apprentissage.")
print("L'utilisation de bibliothèques externes n'est pas autorisée pour le rendu final.")
