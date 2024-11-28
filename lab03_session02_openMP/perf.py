import subprocess


def main():

    output_tab = "num_threads, time(s)\n"

    threads = [1, 2, 3, 4, 5, 6]
    threads.extend(range(10, 110, 10))
    for nb_thread in threads:
        output = subprocess.check_output(
            [
                f"OMP_NUM_THREADS={nb_thread}" "./neural_network",
            ],
        ).decode()

        time = output.strip().split(" ")[-1][:-1]
        time = int(time)
        output_tab += f"{nb_thread},{time}\n"

    print(output_tab)


if __name__ == "__main__":
    main()
