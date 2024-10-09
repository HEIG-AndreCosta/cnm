import matplotlib.pyplot as plt
import os
import subprocess
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

def main(start_size, end_size, increment, tile_size=None, filename=None):
    print("Compiling file")
    os.system("gcc -O0 -o main main.c matrix.c")

    matrix_sizes = list(range(start_size, end_size + 1, increment))
    title = f"Tiled {str(tile_size)}" if tile_size else "Naive"

    l1_usage = []
    l2_hit_rate = []
    execution_times = []

    for matrix_size in tqdm(matrix_sizes, desc="Running Matrix Multiplication", unit="matrix"):
        # Mesure des statistiques du cache
        stats = get_cache_stats("./main", matrix_size, tile_size)
        if stats["L1-loads"] > 0:
            l1_hit_rate = ((stats["L1-loads"] - stats["L1-misses"]) / stats["L1-loads"]) * 100
        else:
            l1_hit_rate = 0

        if stats["L2-loads"] > 0:
            l2_hit_rate_value = ((stats["L2-loads"] - stats["L2-misses"]) / stats["L2-loads"]) * 100
        else:
            l2_hit_rate_value = 0

        l1_usage.append(l1_hit_rate)
        l2_hit_rate.append(l2_hit_rate_value)
        
        # Mesure du temps d'exécution
        execution_time = get_execution_time("./main", matrix_size, tile_size)
        execution_times.append(execution_time)

    # Sauvegarde des graphiques
    folder = "perf_plots"
    if not os.path.exists(folder):
        os.makedirs(folder)
    if filename is None:
        filename = f"plot_start{start_size}_end{end_size}_inc{increment}{'_tiled' + str(tile_size) if tile_size else '_naive'}"

    # Graphique du temps
    fig1, ax1 = plt.subplots()
    ax1.plot(matrix_sizes, execution_times, label="Execution Time")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_xlabel("Matrix Size")
    ax1.set_title("Matrix Multiplication Performance - " + title)
    ax1.legend()
    path_time = os.path.join(folder, f"{filename}_t.svg")
    fig1.savefig(path_time)
    plt.close(fig1)
    print(f"Graph du temps sauvegardé en {path_time}")

    # Graphique L1
    fig2, ax2 = plt.subplots()
    ax2.plot(matrix_sizes, l1_usage, label="L1 Cache Hit %")
    ax2.set_ylabel("Cache Rate (%)")
    ax2.set_xlabel("Matrix Size")
    ax2.set_title("L1 Cache Usage - " + title)
    ax2.legend()
    path_l1 = os.path.join(folder, f"{filename}_l1.svg")
    fig2.savefig(path_l1)
    plt.close(fig2)
    print(f"Graph L1 sauvegardé en {path_l1}")

    # Graphique L2
    fig3, ax3 = plt.subplots()
    ax3.plot(matrix_sizes, l2_hit_rate, label="L2 Cache Hit %")
    ax3.set_ylabel("Cache Rate (%)")
    ax3.set_xlabel("Matrix Size")
    ax3.set_title("L2 Cache Usage - " + title)
    ax3.legend()
    path_l2 = os.path.join(folder, f"{filename}_l2.svg")
    fig3.savefig(path_l2)
    plt.close(fig3)
    print(f"Graph L2 sauvegardé en {path_l2}")

    # Superposition L1 et L2
    fig4, ax4 = plt.subplots()
    ax4.plot(matrix_sizes, l1_usage, label="L1 Cache Hit %")
    ax4.plot(matrix_sizes, l2_hit_rate, label="L2 Cache Hit %")
    ax4.set_ylabel("Cache Rate (%)")
    ax4.set_xlabel("Matrix Size")
    ax4.set_title("Cache Usage - " + title)
    ax4.legend()
    path_l1l2 = os.path.join(folder, f"{filename}_l1l2.svg")
    fig4.savefig(path_l1l2)
    plt.close(fig4)
    print(f"Graph L1 et L2 sauvegardé en {path_l1l2}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Matrix multiplication performance script.")
    
    parser.add_argument("-s", "--start", type=int, required=True, help="Start matrix size.")
    parser.add_argument("-e", "--end", type=int, required=True, help="End matrix size.")
    parser.add_argument("-i", "--increment", type=int, required=True, help="Increment for matrix sizes.")
    parser.add_argument("-t", "--tile", type=int, help="Tile size for tiled multiplication.")
    parser.add_argument("-F", "--file", type=str, help="Base name of the saved files.")

    args = parser.parse_args()

    main(args.start, args.end, args.increment, args.tile, args.file)
