import matplotlib.pyplot as plt
import os
import subprocess
import argparse
from tqdm import tqdm


# Fonction pour récupérer les statistiques du cache avec perf
def get_cache_stats(executable, matrix_size, tile_size=None):
    command = [
        "sudo", "perf", "stat", "-x,", "-e",
        "L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.demand_data_rd_miss,l2_rqsts.all_demand_data_rd"
    ]
    
    if tile_size:
        command += [executable, str(matrix_size), str(tile_size)]
    else:
        command += [executable, str(matrix_size)]
    
    try:
        result = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        result = e.output.decode()

    stats = {"L1-loads": 0, "L1-misses": 0, "L2-misses": 0, "L2-loads": 0}
    
    for line in result.split("\n"):
        parts = line.split(",")
        if len(parts) < 2:
            continue

        if "L1-dcache-loads" in parts[2]:
            try:
                stats["L1-loads"] = int(parts[0].replace(",", ""))
            except ValueError:
                continue
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


# Fonction pour récupérer le temps d'exécution
def get_execution_time(executable, matrix_size, tile_size=None):
    if tile_size:
        result = float(subprocess.check_output([executable, str(matrix_size), str(tile_size)]).decode())
    else:
        result = float(subprocess.check_output([executable, str(matrix_size)]).decode())
    
    return result


def main(start_size, end_size, increment, measure_time=False, measure_cache=False, tile_size=None):
    print("Compiling file")
    os.system("gcc -O0 -o main main.c matrix.c")

    matrix_sizes = list(range(start_size, end_size + 1, increment))
    title = "Tiled" if tile_size else "Naive"

    l1_usage = []
    l2_hit_rate = []
    execution_times = []

    for matrix_size in tqdm(matrix_sizes, desc="Running Matrix Multiplication", unit="matrix"):
        if measure_cache:
            stats = get_cache_stats("./main", matrix_size, tile_size)
            
            if stats["L1-loads"] > 0:
                l1_hit_rate = (stats["L1-loads"] - stats["L1-misses"]) / stats["L1-loads"] * 100
            else:
                l1_hit_rate = 0

            if stats["L2-loads"] > 0:
                l2_hit_rate_value = (stats["L2-loads"] - stats["L2-misses"]) / stats["L2-loads"] * 100
            else:
                l2_hit_rate_value = 0

            l1_usage.append(l1_hit_rate)
            l2_hit_rate.append(l2_hit_rate_value)
        
        if measure_time:
            execution_time = get_execution_time("./main", matrix_size, tile_size)
            execution_times.append(execution_time)

    # Affichage des graphiques
    fig, ax1 = plt.subplots(2 if measure_cache and measure_time else 1, 1, sharex=True)

    if measure_cache:
        ax1[0].plot(matrix_sizes, l1_usage, label="L1 Cache Hit %")
        ax1[0].plot(matrix_sizes, l2_hit_rate, label="L2 Cache Hit %")
        ax1[0].set_ylabel("Cache Rate (%)")
        ax1[0].set_title("Cache Usage - " + title)
        ax1[0].legend()

    if measure_time:
        ax1[1 if measure_cache else 0].plot(matrix_sizes, execution_times, label="Execution Time")
        ax1[1 if measure_cache else 0].set_ylabel("Execution Time (s)")
        ax1[1 if measure_cache else 0].set_xlabel("Matrix Size")
        ax1[1 if measure_cache else 0].set_title("Matrix Multiplication Performance - " + title)
        ax1[1 if measure_cache else 0].legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix multiplication performance script.")
    
    parser.add_argument("-s", "--start", type=int, required=True, help="Start matrix size.")
    parser.add_argument("-e", "--end", type=int, required=True, help="End matrix size.")
    parser.add_argument("-i", "--increment", type=int, required=True, help="Increment for matrix sizes.")
    parser.add_argument("-t", "--tile", type=int, help="Tile size for tiled multiplication.")
    parser.add_argument("-c", "--cache", action="store_true", help="Measure cache usage.")
    parser.add_argument("-T", "--time", action="store_true", help="Measure execution time.")

    args = parser.parse_args()

    main(args.start, args.end, args.increment, args.time, args.cache, args.tile)
