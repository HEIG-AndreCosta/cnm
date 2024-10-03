#!/bin/sh

gcc -o test test.c matrix.c
gcc -o test_tile test_tile.c matrix.c

./test &
./test_tile &

wait

