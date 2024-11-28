import subprocess

import os
from pprint import pprint


def main():

    output_tab = "num_threads, omp_proc_bind, omp_places, time(s)\n"

    threads = [1, 2, 3, 4, 5, 6]
    threads.extend(range(10, 110, 10))
    proc_binds = ["false", "true", "close", "spread", "master"]
    omp_places = "cores"
    for nb_thread in threads:
        for bind in proc_binds:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(nb_thread)
            env["OMP_PROC_BIND"] = bind
            env["OMP_PLACES"] = omp_places
            pprint(env)
            output = subprocess.check_output(
                [
                    "./neural_network",
                ],
                env=env,
            ).decode()

            time = output.strip().split(" ")[-1][:-1]
            time = float(time)
            output_tab += f"{nb_thread}, {bind}, {omp_places}, {time}\n"

    print(output_tab)


if __name__ == "__main__":
    main()
