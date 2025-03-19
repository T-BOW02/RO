import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from matplotlib.patches import Rectangle
import seaborn as sns


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


# Fonction pour évaluer une solution (calcul du makespan et du total weighted tardiness)
def evaluate(permutation, instance_data):
    n_jobs = len(permutation)  # Nombre de pièces
    n_machines = 10  # Nombre de machines fixé à 10

    # Tableau pour stocker les temps de fin de chaque pièce sur chaque machine
    completion_times = np.zeros((n_jobs, n_machines))

    # Traitement de la première pièce
    first_job = permutation[0]
    for j in range(n_machines):
        if j == 0:
            completion_times[0, j] = instance_data[first_job, j + 2]
        else:
            completion_times[0, j] = completion_times[0, j - 1] + instance_data[first_job, j + 2]

    # Traitement des pièces suivantes
    for i in range(1, n_jobs):
        job = permutation[i]
        for j in range(n_machines):
            if j == 0:
                # Sur la première machine, attendre que la pièce précédente soit terminée
                completion_times[i, j] = completion_times[i - 1, j] + instance_data[job, j + 2]
            else:
                # Sur les autres machines, attendre que la machine précédente et la pièce précédente soient libres
                completion_times[i, j] = max(completion_times[i, j - 1], completion_times[i - 1, j]) + instance_data[
                    job, j + 2]

    # Calcul du makespan (temps de fin sur la dernière machine pour la dernière pièce)
    makespan = completion_times[-1, -1]

    # Calcul du total weighted tardiness et des tardiness individuelles
    total_weighted_tardiness = 0
    individual_tardiness = []

    for i in range(n_jobs):
        job = permutation[i]
        completion_time = completion_times[i, -1]  # Temps de fin sur la dernière machine
        priority = instance_data[job, 0]
        deadline = instance_data[job, 1]
        tardiness = max(0, completion_time - deadline)
        weighted_tardiness = tardiness * priority
        total_weighted_tardiness += weighted_tardiness
        individual_tardiness.append((job, tardiness, weighted_tardiness, priority, deadline))

    return makespan, total_weighted_tardiness, completion_times, individual_tardiness


# Fonction pour tracer la frontière Pareto
def plot_pareto_front(pareto_objs, highlight_indices=None):
    makespan_values = [obj[0] for obj in pareto_objs]
    tardiness_values = [obj[1] for obj in pareto_objs]

    plt.figure(figsize=(10, 6))

    # Tracer tous les points
    plt.scatter(makespan_values, tardiness_values, c='blue', marker='o')
    plt.plot(makespan_values, tardiness_values, 'b--', alpha=0.3)

    # Mettre en évidence les points sélectionnés
    if highlight_indices:
        for idx in highlight_indices:
            if 0 <= idx < len(pareto_objs):
                plt.scatter(makespan_values[idx], tardiness_values[idx],
                            c='red', marker='*', s=200, label=f'Solution {idx}')

    plt.xlabel('Makespan')
    plt.ylabel('Total Weighted Tardiness')
    plt.title('Frontière Pareto Optimale')
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, (ms, twd) in enumerate(zip(makespan_values, tardiness_values)):
        plt.annotate(str(i), (ms, twd), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    if highlight_indices:
        plt.legend()

    plt.tight_layout()
    plt.savefig('pareto_front.png')
    plt.show()

    return makespan_values, tardiness_values


# Fonction pour créer un diagramme de Gantt pour une solution spécifique
def plot_gantt_chart(solution_index, permutation, completion_times, instance_data):
    n_jobs = len(permutation)
    n_machines = 10

    # Calculer les temps de début pour chaque opération
    start_times = np.zeros((n_jobs, n_machines))

    for i in range(n_jobs):
        for j in range(n_machines):
            if j == 0 and i == 0:
                start_times[i, j] = 0
            elif j == 0:
                start_times[i, j] = completion_times[i - 1, j]
            elif i == 0:
                start_times[i, j] = completion_times[i, j - 1]
            else:
                start_times[i, j] = max(completion_times[i - 1, j], completion_times[i, j - 1])

    # Créer le diagramme de Gantt
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Colormap pour distinguer les différentes pièces
    colors = plt.cm.jet(np.linspace(0, 1, n_jobs))

    for i in range(n_jobs):
        job = permutation[i]
        for j in range(n_machines):
            duration = instance_data[job, j + 2]
            rect = Rectangle((start_times[i, j], j), duration, 0.8,
                             facecolor=colors[job % len(colors)], edgecolor='black', alpha=0.7)
            ax.add_patch(rect)

            # Ajouter le numéro de la pièce au centre de chaque rectangle
            plt.text(start_times[i, j] + duration / 2, j + 0.4, str(job),
                     ha='center', va='center', color='white', fontweight='bold')

    # Paramètres du graphique
    plt.yticks(range(n_machines), [f'Machine {j + 1}' for j in range(n_machines)])
    plt.xlabel('Temps')
    plt.ylabel('Machines')
    plt.title(f'Diagramme de Gantt - Solution {solution_index}')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Adapter les limites du graphique
    makespan = completion_times[-1, -1]
    plt.xlim(0, makespan * 1.05)
    plt.tight_layout()
    plt.savefig(f'gantt_solution_{solution_index}.png')
    plt.show()


# Fonction pour visualiser les retards des pièces pour une solution
def plot_tardiness_distribution(solution_index, individual_tardiness):
    # Trier par pièce
    individual_tardiness.sort(key=lambda x: x[0])

    jobs = [item[0] for item in individual_tardiness]
    tardiness = [item[1] for item in individual_tardiness]
    weighted_tardiness = [item[2] for item in individual_tardiness]

    # Créer la figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Distribution des retards
    ax1.hist(tardiness, bins=30, alpha=0.7, color='blue')
    ax1.set_xlabel('Retard')
    ax1.set_ylabel('Nombre de pièces')
    ax1.set_title('Distribution des retards')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Retards par pièce (top 20 des plus retardées)
    delayed_jobs = [j for j, t in zip(jobs, tardiness) if t > 0]
    delayed_tardiness = [t for t in tardiness if t > 0]

    if delayed_jobs:
        # Trier par retard décroissant
        sorted_indices = np.argsort(delayed_tardiness)[::-1][:20]  # Top 20
        top_jobs = [delayed_jobs[i] for i in sorted_indices]
        top_tardiness = [delayed_tardiness[i] for i in sorted_indices]

        ax2.bar(range(len(top_jobs)), top_tardiness, color='red')
        ax2.set_xlabel('Indice de la pièce')
        ax2.set_ylabel('Retard')
        ax2.set_title('Top 20 des pièces les plus retardées')
        ax2.set_xticks(range(len(top_jobs)))
        ax2.set_xticklabels(top_jobs, rotation=90)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, 'Aucune pièce en retard', ha='center', va='center', fontsize=12)
        ax2.set_title('Pièces en retard')

    plt.tight_layout()
    plt.savefig(f'tardiness_solution_{solution_index}.png')
    plt.show()


# Fonction pour visualiser l'utilisation des machines
def plot_machine_utilization(solution_index, permutation, completion_times, instance_data):
    n_jobs = len(permutation)
    n_machines = 10

    # Calculer les temps de début pour chaque opération
    start_times = np.zeros((n_jobs, n_machines))

    for i in range(n_jobs):
        for j in range(n_machines):
            if j == 0 and i == 0:
                start_times[i, j] = 0
            elif j == 0:
                start_times[i, j] = completion_times[i - 1, j]
            elif i == 0:
                start_times[i, j] = completion_times[i, j - 1]
            else:
                start_times[i, j] = max(completion_times[i - 1, j], completion_times[i, j - 1])

    # Calculer les durées de chaque opération
    durations = np.zeros((n_jobs, n_machines))
    for i in range(n_jobs):
        job = permutation[i]
        for j in range(n_machines):
            durations[i, j] = instance_data[job, j + 2]

    # Créer une matrice pour la heatmap des utilisation des machines
    makespan = int(completion_times[-1, -1])
    utilization = np.zeros((n_machines, makespan + 1))

    for i in range(n_jobs):
        for j in range(n_machines):
            start = int(start_times[i, j])
            end = int(start_times[i, j] + durations[i, j])
            utilization[j, start:end] = 1

    # Créer la heatmap
    plt.figure(figsize=(15, 6))
    sns.heatmap(utilization, cmap="Blues", cbar=False,
                yticklabels=[f'Machine {j + 1}' for j in range(n_machines)])
    plt.xlabel('Temps')
    plt.ylabel('Machines')
    plt.title(f'Utilisation des machines - Solution {solution_index}')

    # Adapter l'axe x pour qu'il ne soit pas trop dense
    step = max(1, makespan // 20)
    plt.xticks(np.arange(0, makespan + 1, step))

    plt.tight_layout()
    plt.savefig(f'machine_utilization_{solution_index}.png')
    plt.show()


# Fonction pour comparer deux solutions
def compare_solutions(solution1_idx, solution2_idx, permutations, completion_times_list, individual_tardiness_list,
                      pareto_objs):
    # Obtenir les objectifs
    makespan1, tard1 = pareto_objs[solution1_idx]
    makespan2, tard2 = pareto_objs[solution2_idx]

    # Créer la figure
    plt.figure(figsize=(12, 6))

    # Comparaison des objectifs
    labels = ['Solution ' + str(solution1_idx), 'Solution ' + str(solution2_idx)]
    makespan_values = [makespan1, makespan2]
    tard_values = [tard1, tard2]

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, makespan_values, width, label='Makespan')
    plt.bar(x + width / 2, tard_values, width, label='Total Weighted Tardiness')

    plt.ylabel('Valeur')
    plt.title('Comparaison des objectifs')
    plt.xticks(x, labels)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'compare_solutions_{solution1_idx}_{solution2_idx}.png')
    plt.show()


# Fonction principale
def main():
    parser = argparse.ArgumentParser(description='Visualisation des solutions du problème de Flowshop Bi-objectif')
    parser.add_argument('--instance', type=str, default='instance.csv', help='Fichier d\'instance')
    parser.add_argument('--solutions', type=str, required=True, help='Fichier CSV contenant les solutions Pareto')
    parser.add_argument('--gantt', type=int, default=None,
                        help='Indice de la solution à visualiser avec un diagramme de Gantt')
    parser.add_argument('--tardiness', type=int, default=None,
                        help='Indice de la solution pour visualiser la distribution des retards')
    parser.add_argument('--compare', nargs=2, type=int, default=None, help='Indices des deux solutions à comparer')
    parser.add_argument('--highlight', nargs='+', type=int, default=None,
                        help='Indices des solutions à mettre en évidence sur la frontière Pareto')
    parser.add_argument('--utilization', type=int, default=None,
                        help='Indice de la solution pour visualiser l\'utilisation des machines')

    args = parser.parse_args()

    # Lire l'instance et les solutions
    instance_data = read_instance(args.instance)
    pareto_solutions = read_pareto_solutions(args.solutions)

    print(f"Instance: {args.instance}")
    print(f"Solutions: {args.solutions}")
    print(f"Nombre de solutions: {len(pareto_solutions)}")

    # Évaluer chaque solution
    pareto_objs = []
    completion_times_list = []
    individual_tardiness_list = []

    for i, solution in enumerate(pareto_solutions):
        makespan, tardiness, completion_times, individual_tardiness = evaluate(solution, instance_data)
        pareto_objs.append((makespan, tardiness))
        completion_times_list.append(completion_times)
        individual_tardiness_list.append(individual_tardiness)
        print(f"Solution {i}: Makespan = {makespan:.2f}, Total Weighted Tardiness = {tardiness:.2f}")

    # Tracer la frontière Pareto
    makespan_values, tardiness_values = plot_pareto_front(pareto_objs, args.highlight)

    # Visualiser le diagramme de Gantt si demandé
    if args.gantt is not None and 0 <= args.gantt < len(pareto_solutions):
        solution_index = args.gantt
        print(f"\nGénération du diagramme de Gantt pour la solution {solution_index}...")
        plot_gantt_chart(solution_index, pareto_solutions[solution_index],
                         completion_times_list[solution_index], instance_data)

    # Visualiser la distribution des retards si demandé
    if args.tardiness is not None and 0 <= args.tardiness < len(pareto_solutions):
        solution_index = args.tardiness
        print(f"\nGénération de la distribution des retards pour la solution {solution_index}...")
        plot_tardiness_distribution(solution_index, individual_tardiness_list[solution_index])

    # Comparer deux solutions si demandé
    if args.compare is not None and len(args.compare) == 2:
        solution1_idx, solution2_idx = args.compare
        if 0 <= solution1_idx < len(pareto_solutions) and 0 <= solution2_idx < len(pareto_solutions):
            print(f"\nComparaison des solutions {solution1_idx} et {solution2_idx}...")
            compare_solutions(solution1_idx, solution2_idx, pareto_solutions,
                              completion_times_list, individual_tardiness_list, pareto_objs)

    # Visualiser l'utilisation des machines si demandé
    if args.utilization is not None and 0 <= args.utilization < len(pareto_solutions):
        solution_index = args.utilization
        print(f"\nGénération de la heatmap d'utilisation des machines pour la solution {solution_index}...")
        plot_machine_utilization(solution_index, pareto_solutions[solution_index],
                                 completion_times_list[solution_index], instance_data)


if __name__ == "__main__":
    main()
