import matplotlib.pyplot as plt
import os
import subprocess
import argparse
from tqdm import tqdm  # Importation de tqdm pour la barre de progression


def main(start_size, end_size, increment, tile_size=None):
    print("Compiling file")
    os.system("gcc -O0 -o main main.c matrix.c")

    print("Running the implementation")

    matrix_sizes = list(range(start_size, end_size + 1, increment))
    title = "Tiled" if tile_size else "Naive"

    if tile_size:
        print(f"Running the tiled implementation with tile size: {tile_size}")
        # Utilisation de tqdm pour afficher la progression
        outputs_tiling = [
            float(subprocess.check_output(["./main", str(matrix_size), str(tile_size)]).decode())
            for matrix_size in tqdm(matrix_sizes, desc="Tiling Progress", unit="matrix")
        ]
        plt.plot(matrix_sizes, outputs_tiling, label="Tiling")
    else:
        print("Running the naive implementation")
        # Utilisation de tqdm pour afficher la progression
        outputs_naive = [
            float(subprocess.check_output(["./main", str(matrix_size)]).decode())
            for matrix_size in tqdm(matrix_sizes, desc="Naive Progress", unit="matrix")
        ]
        plt.plot(matrix_sizes, outputs_naive, label="Naive")

    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Matrix Multiplication Performance - " + title)
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
