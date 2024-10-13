import subprocess
import os


def images_paths():

    return [os.path.join("images", x) for x in os.listdir("images")]


def main():

    targets = [
        "edge_detection_no_unroll",
        "edge_detection_manual_unroll",
        "edge_detection_compilero0_unroll",
        "edge_detection_compilero1_unroll",
        "edge_detection_compilero2_unroll",
    ]
    images = images_paths()

    procs = {
        image: [
            {
                "target": target,
                "process": subprocess.Popen(
                    [f"./{target}", image],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                ),
            }
            for target in targets
        ]
        for image in images
    }
    print(procs)

    print(
        "| Image  | Rows    | Columns | No Loop Unrolling | Conv. time (SW loop unrolling)| Conv. time (compiler -O0)| Conv. time (compiler -O1)| Conv. time (compiler -O2)|"
    )
    print(
        "|--------|---------|---------|-------------------|-------------------------------|--------------------------|--------------------------|--------------------------|"
    )
    for image_name, processes in procs.items():

        print(f"|{image_name}|")
        for i, proc in enumerate(processes):
            output, _ = proc["process"].communicate()
            rows, cols, time = output.decode().split(", ")
            if i == 0:
                print(f"{rows}|{cols}|{time}|", end="")
            else:
                print(f"{time}|")


if __name__ == "__main__":
    main()
