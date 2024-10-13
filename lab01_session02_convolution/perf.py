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

    output_tab = "| Image  | Rows    | Columns | No Loop Unrolling | Conv. time (SW loop unrolling)| Conv. time (compiler -O0)| Conv. time (compiler -O1)| Conv. time (compiler -O2)|\n"
    output_tab += "|--------|---------|---------|-------------------|-------------------------------|--------------------------|--------------------------|--------------------------|\n"
    for image in images:
        output_tab += f"|{image}|"
        for i, target in enumerate(targets):
            output = subprocess.check_output(
                [f"./{target}", image],
            ).decode()
            print(f"Running {[f'./{target}', image]}")

            rows, cols, time = output.strip().split(", ")
            if i == 0:
                output_tab += f"{rows}|{cols}|{time}|"
            else:
                output_tab += f"{time}|"
        output_tab += "\n"

    print(output_tab)


if __name__ == "__main__":
    main()
