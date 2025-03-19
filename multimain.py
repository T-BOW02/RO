import numpy as np
import random
import csv
import multiprocessing as mp
import time
from copy import deepcopy


# Lecture des données d'instance
def read_instance(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(val) for val in row])
    return np.array(data)


# Calcul du makespan et du total weighted tardiness pour une permutation donnée
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

    # Calcul du total weighted tardiness
    total_weighted_tardiness = 0
    for i in range(n_jobs):
        job = permutation[i]
        completion_time = completion_times[i, -1]  # Temps de fin sur la dernière machine
        priority = instance_data[job, 0]
        deadline = instance_data[job, 1]
        tardiness = max(0, completion_time - deadline)
        total_weighted_tardiness += tardiness * priority

    return makespan, total_weighted_tardiness


# Vérification de la domination entre deux solutions
def dominates(obj1, obj2):
    return (obj1[0] <= obj2[0] and obj1[1] < obj2[1]) or (obj1[0] < obj2[0] and obj1[1] <= obj2[1])


# Génération du voisinage par échange de deux positions
def get_swap_neighborhood(solution, tabu_list, size=None):
    n = len(solution)
    neighbors = []
    moves = []

    # Générer tous les mouvements possibles (échanges)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Si une taille est spécifiée, échantillonner aléatoirement
    if size and size < len(all_pairs):
        sampled_pairs = random.sample(all_pairs, size)
    else:
        sampled_pairs = all_pairs

    for i, j in sampled_pairs:
        # Vérifier si le mouvement est tabou
        if (i, j) not in tabu_list and (j, i) not in tabu_list:
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
            moves.append((i, j))

    return neighbors, moves


# Génération du voisinage par insertion
def get_insertion_neighborhood(solution, tabu_list, size=None):
    n = len(solution)
    neighbors = []
    moves = []

    # Générer tous les mouvements possibles (insertions)
    all_moves = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Si une taille est spécifiée, échantillonner aléatoirement
    if size and size < len(all_moves):
        sampled_moves = random.sample(all_moves, size)
    else:
        sampled_moves = all_moves

    for i, j in sampled_moves:
        # Vérifier si le mouvement est tabou
        if (i, j) not in tabu_list:
            neighbor = solution.copy()
            # Retirer l'élément à la position i
            value = neighbor.pop(i)
            # Insérer l'élément à la position j (ou j-1 si j > i)
            neighbor.insert(j if j < i else j - 1, value)
            neighbors.append(neighbor)
            moves.append((i, j))

    return neighbors, moves


# Mise à jour de l'archive Pareto
def update_pareto_archive(archive, archive_objs, new_solution, new_obj):
    # Vérifier si la nouvelle solution est dominée par une solution existante
    for obj in archive_objs:
        if dominates(obj, new_obj):
            return False, archive, archive_objs

    # Supprimer les solutions dominées par la nouvelle solution
    non_dominated = []
    non_dominated_objs = []
    for i, obj in enumerate(archive_objs):
        if not dominates(new_obj, obj):
            non_dominated.append(archive[i])
            non_dominated_objs.append(obj)

    # Ajouter la nouvelle solution
    non_dominated.append(new_solution.copy())
    non_dominated_objs.append(new_obj)

    return True, non_dominated, non_dominated_objs


# Recherche tabou multi-objectif
def multi_objective_tabu_search(instance_data, max_iterations=1000, tabu_tenure=20, neighborhood_size=50):
    n_jobs = len(instance_data)

    # Solution initiale aléatoire
    current_solution = list(range(n_jobs))
    random.shuffle(current_solution)

    # Évaluation de la solution initiale
    current_obj = evaluate(current_solution, instance_data)

    # Initialisation de l'archive Pareto
    pareto_archive = [current_solution.copy()]
    pareto_objs = [current_obj]

    # Initialisation de la liste tabou
    tabu_list = []

    # Compteur d'itérations sans amélioration
    iterations_without_improvement = 0

    for it in range(max_iterations):
        if it % 100 == 0:
            print(f"Itération {it + 1}/{max_iterations}, taille archive: {len(pareto_archive)}")

        # Alternance entre les types de voisinage
        if it % 2 == 0:
            neighbors, moves = get_swap_neighborhood(current_solution, tabu_list, neighborhood_size)
        else:
            neighbors, moves = get_insertion_neighborhood(current_solution, tabu_list, neighborhood_size)

        # Si pas de voisins valides, passer à l'itération suivante
        if not neighbors:
            iterations_without_improvement += 1
            continue

        # Évaluation des voisins
        neighbor_objs = [evaluate(neighbor, instance_data) for neighbor in neighbors]

        # Identifier les voisins non dominés
        non_dominated_indices = []
        for i, obj in enumerate(neighbor_objs):
            if not any(dominates(other_obj, obj) for j, other_obj in enumerate(neighbor_objs) if j != i):
                non_dominated_indices.append(i)

        # Mise à jour de l'archive Pareto
        improvement = False
        for i in non_dominated_indices:
            added, pareto_archive, pareto_objs = update_pareto_archive(pareto_archive, pareto_objs, neighbors[i],
                                                                       neighbor_objs[i])
            if added:
                improvement = True

        # Sélection de la prochaine solution parmi les non dominées
        if non_dominated_indices:
            # Choisir aléatoirement parmi les non dominés
            idx = random.choice(non_dominated_indices)
            current_solution = neighbors[idx]
            current_obj = neighbor_objs[idx]

            # Mise à jour de la liste tabou
            move = moves[idx]
            tabu_list.append(move)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

        # Mise à jour du compteur d'itérations sans amélioration
        if improvement:
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        # Diversification si on reste bloqué
        if iterations_without_improvement > 100:
            print("Diversification...")
            current_solution = list(range(n_jobs))
            random.shuffle(current_solution)
            current_obj = evaluate(current_solution, instance_data)
            tabu_list = []
            iterations_without_improvement = 0

    # Trier les solutions Pareto par makespan croissant
    sorted_indices = sorted(range(len(pareto_objs)), key=lambda i: pareto_objs[i][0])
    pareto_archive = [pareto_archive[i] for i in sorted_indices]
    pareto_objs = [pareto_objs[i] for i in sorted_indices]

    return pareto_archive, pareto_objs


# Initialiser une solution avec une heuristique pour le makespan (NEH modifié)
def neh_heuristic(instance_data):
    n_jobs = len(instance_data)

    # Calculer le temps total de traitement pour chaque pièce
    processing_times = np.sum(instance_data[:, 2:], axis=1)

    # Trier les pièces par temps de traitement décroissant
    sorted_jobs = sorted(range(n_jobs), key=lambda i: processing_times[i], reverse=True)

    # Construire une solution avec NEH
    sequence = [sorted_jobs[0]]

    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_makespan = float('inf')
        best_position = 0

        # Essayer d'insérer le job à chaque position possible
        for j in range(len(sequence) + 1):
            # Créer une nouvelle séquence avec le job inséré à la position j
            new_sequence = sequence.copy()
            new_sequence.insert(j, job)

            # Évaluer le makespan de cette séquence
            makespan, _ = evaluate(new_sequence, instance_data)

            if makespan < best_makespan:
                best_makespan = makespan
                best_position = j

        # Insérer le job à la meilleure position
        sequence.insert(best_position, job)

    return sequence


# Initialiser une solution avec une heuristique pour le total weighted tardiness (WSPT modifié)
def wspt_heuristic(instance_data):
    n_jobs = len(instance_data)

    # Trier les pièces par ratio (priorité / temps de traitement) décroissant
    sorted_jobs = sorted(range(n_jobs),
                         key=lambda i: instance_data[i, 0] / np.sum(instance_data[i, 2:]),
                         reverse=True)

    return sorted_jobs


# Fonction pour traiter un seul démarrage (à placer avant multi_start_search)
def process_single_start(start_id, n_starts, instance_data, iterations_per_start):
    print(f"Démarrage {start_id + 1}/{n_starts}")

    # Initialisation aléatoire
    start_solution = list(range(len(instance_data)))
    random.shuffle(start_solution)

    # Exécution de la recherche tabou
    pareto_front, pareto_objs = multi_objective_tabu_search(
        instance_data,
        max_iterations=iterations_per_start,
        tabu_tenure=20,
        neighborhood_size=50
    )

    return pareto_front, pareto_objs


# Recherche multi-objectif avec plusieurs points de départ
def multi_start_search(instance_data, n_starts=10, iterations_per_start=500):
    import multiprocessing as mp

    # Initialisation de l'archive Pareto globale
    global_archive = []
    global_objs = []

    # Point de départ 1: Heuristique NEH pour minimiser le makespan
    neh_solution = neh_heuristic(instance_data)
    neh_obj = evaluate(neh_solution, instance_data)
    global_archive.append(neh_solution)
    global_objs.append(neh_obj)

    # Point de départ 2: Heuristique WSPT pour minimiser le total weighted tardiness
    wspt_solution = wspt_heuristic(instance_data)
    wspt_obj = evaluate(wspt_solution, instance_data)
    _, global_archive, global_objs = update_pareto_archive(global_archive, global_objs, wspt_solution, wspt_obj)

    # Préparer les arguments pour chaque processus
    args = [(start_id, n_starts, instance_data, iterations_per_start) for start_id in range(n_starts)]

    # Exécution parallèle avec multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        all_results = pool.starmap(process_single_start, args)

    # Mise à jour de l'archive globale avec tous les résultats
    for pareto_front, pareto_objs in all_results:
        for solution, obj in zip(pareto_front, pareto_objs):
            _, global_archive, global_objs = update_pareto_archive(global_archive, global_objs, solution, obj)

    # Trier les solutions Pareto par makespan croissant
    sorted_indices = sorted(range(len(global_objs)), key=lambda i: global_objs[i][0])
    global_archive = [global_archive[i] for i in sorted_indices]
    global_objs = [global_objs[i] for i in sorted_indices]

    return global_archive, global_objs


# Sauvegarde des solutions Pareto dans un fichier CSV
def save_pareto_front(pareto_front, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for solution in pareto_front:
            writer.writerow(solution)
    print(f"Frontière Pareto sauvegardée dans {filename}")


# Fonction principale
def main():
    # Lecture de l'instance
    instance_data = read_instance("instance.csv")

    # Exécution de la recherche multi-départ
    pareto_front, pareto_objectives = multi_start_search(
        instance_data,
        n_starts=56,
        iterations_per_start=5000
    )

    # Affichage des résultats
    print("Nombre de solutions dans la frontière Pareto:", len(pareto_front))
    print("Objectifs (Makespan, Total Weighted Tardiness):")
    for obj in pareto_objectives:
        print(obj)

    # Sauvegarde des résultats
    save_pareto_front(pareto_front, "Marchand_Robin_Marlier_Thibault.csv")


if __name__ == "__main__":
    main()