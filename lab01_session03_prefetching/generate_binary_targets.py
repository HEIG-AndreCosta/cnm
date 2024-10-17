import os


def main():

    for i in range(0, 4):
        os.system(
            f"g++ -pg -O{i} -o target_O{i}_prefetch -DDO_PREFETCH ./convolution_prefetching.cpp"
        )
        os.system(f"g++ -pg -O{i}  -o target_O{i} ./convolution_prefetching.cpp")


if __name__ == "__main__":
    main()
