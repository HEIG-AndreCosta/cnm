import subprocess
import os


def create_if_not_exists(path):
    os.makedirs(path, exist_ok=True)


def images_paths():
    return [os.path.join("images", x) for x in os.listdir("images")]


def main():

    targets = [
        "edge_detection_no_unroll",
        "edge_detection_manualo0_unroll",
        "edge_detection_manualo1_unroll",
        "edge_detection_manualo2_unroll",
        "edge_detection_compilero0_unroll",
        "edge_detection_compilero1_unroll",
        "edge_detection_compilero2_unroll",
    ]
    images = images_paths()

    output_tab = "| Image  | Rows    | Columns | No Loop Unrolling | Conv. time (SW loop unrolling) (-O0)| Conv. time (SW loop unrolling) (-O1)| Conv. time (SW loop unrolling) (-O2)| Conv. time (compiler -O0)| Conv. time (compiler -O1)| Conv. time (compiler -O2)|\n"
    output_tab += "|--------|-|-|---------|---------|-------------------|-------------------------------|--------------------------|--------------------------|--------------------------|\n"
    base_time = 0

    for image in images:
        output_tab += f"|{image}|"
        for i, target in enumerate(targets):

            image_name = os.path.basename(image)
            output_folder = os.path.join("output", image_name, target)
            create_if_not_exists(output_folder)
            output_path = os.path.join(output_folder, image_name)
            output = subprocess.check_output(
                [
                    f"./{target}",
                    image,
                    output_path,
                ],
            ).decode()

            rows, cols, time = output.strip().split(", ")
            time = int(time)
            if i == 0:
                base_time = time
                output_tab += f"{rows}|{cols}|{time} (+0)|"
            else:
                output_tab += f"{time} ({(time - base_time):+})|"
        output_tab += "\n"

    print(output_tab)


if __name__ == "__main__":
    main()
