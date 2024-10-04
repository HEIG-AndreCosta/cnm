import matplotlib.pyplot as plt
import os
import subprocess


def main():
    print("Compiling file")
    os.system("gcc -O0 -o main main.c matrix.c")

    print("Running the naive implementation")

    matrix_sizes = [10, 50, 100, 250, 300, 500]

    outputs_naive = [
        float(subprocess.check_output(["./main", str(matrix_size)]).decode())
        for matrix_size in matrix_sizes
    ]

    plt.plot(matrix_sizes, outputs_naive)

    plt.show()
    outputs_tiling = [
        float(subprocess.check_output(["./main", str(matrix_size), str(5)]).decode())
        for matrix_size in matrix_sizes
    ]
    plt.plot(matrix_sizes, outputs_tiling)

    plt.show()


if __name__ == "__main__":
    main()
