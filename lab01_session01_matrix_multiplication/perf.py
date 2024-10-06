import matplotlib.pyplot as plt
import os
import subprocess
import argparse
from tqdm import tqdm


# Fonction pour récupérer les statistiques du cache avec perf
def get_cache_stats(executable, matrix_size, tile_size=None):
    command = [
        "sudo", "perf", "stat", "-x,", "-e",
        "l1d_cache,l1d_cache_lmiss_rd,l2d_cache,l2d_cache_lmiss_rd"
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
        if "l1d_cache_lmiss_rd" in parts[2]:
            try:
                stats["L1-misses"] = int(parts[0].replace(",", ""))
            except ValueError:
                continue
        elif "l1d_cache" in parts[2]:
            try:
                stats["L1-loads"] = int(parts[0].replace(",", ""))
            except ValueError:
                continue
        elif "l2d_cache_lmiss_rd" in parts[2]:
            try:
                stats["L2-misses"] = int(parts[0].replace(",", ""))
            except ValueError:
                continue
        elif "l2d_cache" in parts[2]:
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


def main(start_size, end_size, increment, measure_time=False, measure_cache=False, tile_size=None, save=False, filename=None):
    print("Compiling file")
    os.system("gcc -O0 -o main main.c matrix.c")

    matrix_sizes = list(range(start_size, end_size + 1, increment))
    title = f"Tiled {str(tile_size)}" if tile_size else "Naive"

    l1_usage = []
    l2_hit_rate = []
    execution_times = []

    for matrix_size in tqdm(matrix_sizes, desc="Running Matrix Multiplication", unit="matrix"):
        if measure_cache:
            stats = get_cache_stats("./main", matrix_size, tile_size) 
            if stats["L1-loads"] > 0:
                l1_hit_rate = ((stats["L1-loads"] - stats["L1-misses"]) / stats["L1-loads"]) * 100
            else:
                l1_hit_rate = 0

            if stats["L2-loads"] > 0:
                l2_hit_rate_value = ((stats["L2-loads"] - stats["L2-misses"]) / stats["L2-loads"] )* 100
            else:
                l2_hit_rate_value = 0

            l1_usage.append(l1_hit_rate)
            l2_hit_rate.append(l2_hit_rate_value)
        
        if measure_time:
            execution_time = get_execution_time("./main", matrix_size, tile_size)
            execution_times.append(execution_time)

    # Affichage des graphiques
    if measure_cache and measure_time:
        fig, ax1 = plt.subplots(2, 1, sharex=True)
    else:
        fig, ax1 = plt.subplots()

    # S'assurer que ax1 est bien une liste d'axes si on a deux graphiques
    if measure_cache and measure_time:
        ax_cache = ax1[0]
        ax_time = ax1[1]
    else:
        ax_cache = ax1 if measure_cache else None
        ax_time = ax1 if measure_time else None

    if measure_cache:
        ax_cache.plot(matrix_sizes, l1_usage, label="L1 Cache Hit %")
        ax_cache.plot(matrix_sizes, l2_hit_rate, label="L2 Cache Hit %")
        ax_cache.set_ylabel("Cache Rate (%)")
        ax_cache.set_title("Cache Usage - " + title)
        ax_cache.legend()

    if measure_time:
        ax_time.plot(matrix_sizes, execution_times, label="Execution Time")
        ax_time.set_ylabel("Execution Time (s)")
        ax_time.set_xlabel("Matrix Size")
        ax_time.set_title("Matrix Multiplication Performance - " + title)
        ax_time.legend()

    # Sauvegarde du graphique en SVG si l'option --save est activée
    if save:
        folder = "perf_plots"
        if not os.path.exists(folder):
            os.makedirs(folder)
        if filename is None:
            filename = f"plot_start{start_size}_end{end_size}_inc{increment}{'_tiled' + str(tile_size) if tile_size else '_naive'}.svg"

        path = os.path.join(folder, filename)
        plt.savefig(path)
        print(f"Graph saved as {filename}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix multiplication performance script.")
    
    parser.add_argument("-s", "--start", type=int, required=True, help="Start matrix size.")
    parser.add_argument("-e", "--end", type=int, required=True, help="End matrix size.")
    parser.add_argument("-i", "--increment", type=int, required=True, help="Increment for matrix sizes.")
    parser.add_argument("-t", "--tile", type=int, help="Tile size for tiled multiplication.")
    parser.add_argument("-c", "--cache", action="store_true", help="Measure cache usage.")
    parser.add_argument("-T", "--time", action="store_true", help="Measure execution time.")
    parser.add_argument("-S", "--save", action="store_true", help="Save the plot as an SVG file.")
    parser.add_argument("-F", "--file",type=str, help="Name of the saved file.")

    args = parser.parse_args()

    main(args.start, args.end, args.increment, args.time, args.cache, args.tile, args.save, args.file)
