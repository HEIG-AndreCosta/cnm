import subprocess
import os


def images_paths():

    return [os.path.join("images", x) for x in os.listdir("images")]


def main():

    targets = [
        "edge_detection_no_unroll",
        "edge_detection_manual_unroll",
        "edge_detection_compiler_unroll",
    ]
    images = images_paths()

    procs = [
        {
            "image": image,
            "target": target,
            "process": subprocess.Popen(
                [target, image],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            ),
        }
        for image, target in zip(images, targets)
    ]
    for process in procs:
        process["process"].wait()
        process["output"] = float(process["process"].stdout.decode())

    for process in procs:
        target = process["target"]
        image = process["image"]
        output = process["output"]
        rows, cols, time = output.split(", ")

        print(f"{target} {image} rows: {rows} cols: {cols} time:{time}")


if __name__ == "__main__":
    main()
