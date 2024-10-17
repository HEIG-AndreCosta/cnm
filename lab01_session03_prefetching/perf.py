import subprocess
import os


def main():

    targets = [f"./target_O{i}" for i in range(0, 4)]

    for i in range(0, 4):
        os.system(f"g++ -O{i}  -o target_O{i} ./convolution_prefetching.cpp")

    for target in targets:
        print(f"Test {target}")
        print(
            "|Matrix Size|Prefetch Offset|Time With Prefetch|Time Without Prefetch (us)|"
        )
        print("|-|-|-:|-:|")
        for i in range(2, 20):
            matrix_size = 2**i
            for j in range(0, 8):
                offset = 2**j

                output_lines = (
                    subprocess.check_output([target, str(matrix_size), str(offset)])
                    .decode()
                    .strip()
                    .split("\n")
                )
                time_with_prefetch = float(output_lines[0].split(" ")[-2]) * 1e6
                time_without_prefetch = float(output_lines[1].split(" ")[-2]) * 1e6
                print(
                    f"|{matrix_size}|{offset}|{time_with_prefetch}|{time_without_prefetch}|"
                )


if __name__ == "__main__":
    main()
