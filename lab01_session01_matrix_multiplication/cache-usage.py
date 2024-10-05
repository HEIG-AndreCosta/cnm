import matplotlib.pyplot as plt
import os
import subprocess
import argparse
from tqdm import tqdm  # Importation de tqdm pour la barre de progression


# Fonction pour récupérer les statistiques du cache avec perf
def get_cache_stats(executable, matrix_size, tile_size=None):
    # Commande perf pour récupérer les stats L1 et L2
    command = [
        "sudo", "perf", "stat", "-x,", "-e", 
        "L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.demand_data_rd_miss,l2_rqsts.all_demand_data_rd"
    ]
    
    if tile_size:
        command += [executable, str(matrix_size), str(tile_size)]
    else:
        command += [executable, str(matrix_size)]
    
    # Exécuter la commande et capturer la sortie
    try:
        result = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        result = e.output.decode()

    # Initialiser les stats
    stats = {"L1-loads": 0, "L1-misses": 0, "L2-misses": 0, "L2-loads": 0}
    
    for line in result.split("\n"):
        parts = line.split(",")
        if len(parts) < 2:
            continue  # Ignore les lignes incorrectes

        # Associer les valeurs aux stats
        if "L1-dcache-loads" in parts[2]:
            try:
                stats["L1-loads"] = int(parts[0].replace(",", ""))
            except ValueError:
                continue  # Ignore si la conversion échoue
        elif "L1-dcache-load-misses" in parts[2]:
            try:
                stats["L1-misses"] = int(parts[0].replace(",", ""))
            except ValueError:
                continue
        elif "l2_rqsts.demand_data_rd_miss" in parts[2]:
            try:
                stats["L2-misses"] = int(parts[0].replace(",", ""))
            except ValueError:
                continue
        elif "l2_rqsts.all_demand_data_rd" in parts[2]:
            try:
                stats["L2-loads"] = int(parts[0].replace(",", ""))
            except ValueError:
                continue
    
    return stats


def main(start_size, end_size, increment, tile_size=None):
    print("Compiling file")
    os.system("gcc -O0 -o main main.c matrix.c")

    print("Running the implementation")

    matrix_sizes = list(range(start_size, end_size + 1, increment))
    title = "Tiled" if tile_size else "Naive"

    l1_usage = []
    l2_miss_rate = []

    for matrix_size in tqdm(matrix_sizes, desc="Running Matrix Multiplication", unit="matrix"):
        stats = get_cache_stats("./main", matrix_size, tile_size)

        # Calcul du pourcentage d'utilisation du cache L1 et L2 miss rate
        if stats["L1-loads"] > 0:
            l1_hit_rate = (stats["L1-loads"] - stats["L1-misses"]) / stats["L1-loads"] * 100
        else:
            l1_hit_rate = 0

        if stats["L2-loads"] > 0:
            l2_miss_rate_value = stats["L2-misses"] / stats["L2-loads"] * 100
        else:
            l2_miss_rate_value = 0

        l1_usage.append(l1_hit_rate)
        l2_miss_rate.append(l2_miss_rate_value)

    # Plot the results for L1 usage and L2 miss rate
    plt.plot(matrix_sizes, l1_usage, label="L1 Cache Hit %")
    plt.plot(matrix_sizes, l2_miss_rate, label="L2 Cache Miss %")
    
    plt.xlabel("Matrix Size")
    plt.ylabel("Cache Rate (%)")
    plt.title("Cache Usage - " + title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix multiplication performance script.")
    
    parser.add_argument("-s", "--start", type=int, required=True, help="Start matrix size.")
    parser.add_argument("-e", "--end", type=int, required=True, help="End matrix size.")
    parser.add_argument("-i", "--increment", type=int, required=True, help="Increment for matrix sizes.")
    parser.add_argument("-t", "--tile", type=int, help="Tile size for tiled multiplication.")

    args = parser.parse_args()

    main(args.start, args.end, args.increment, args.tile)
