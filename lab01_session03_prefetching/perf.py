import subprocess


def main():

    print("|Offset|Time With Prefetch|Time Without Prefetch (us)|")
    print("|-|-|-|")
    for i in range(0, 10):
        offset = 2**i
        output_lines = (
            subprocess.check_output(["./target", str(offset)])
            .decode()
            .strip()
            .split("\n")
        )
        time_with_prefetch = float(output_lines[0].split(" ")[-2]) * 1e6
        time_without_prefetch = float(output_lines[1].split(" ")[-2]) * 1e6
        print(f"|{offset}|{time_with_prefetch}|{time_without_prefetch}|")


if __name__ == "__main__":
    main()
