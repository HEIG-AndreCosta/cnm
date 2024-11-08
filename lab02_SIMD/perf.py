import subprocess
import os


def create_if_not_exists(path):
    os.makedirs(path, exist_ok=True)


def images_paths():
    return [os.path.join("images", x) for x in os.listdir("images")]


def main():

    targets = [
        "edge_detection_simd",
    ]
    images = images_paths()

    output_tab = "Image,Rows,Columns,Conv. Time\n"

    for image in images:
        output_tab += f"|{image}|"
        for target in targets:
            output = subprocess.check_output(
                [
                    f"./{target}",
                    image,
                ],
            ).decode()

            rows, cols, time = output.strip().split(", ")
            time = int(time)
            output_tab += f"{rows},{cols},{time}"
        output_tab += "\n"

    print(output_tab)


if __name__ == "__main__":
    main()
