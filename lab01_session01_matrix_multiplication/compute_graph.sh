#/bin/bash

# set env

source /venv/bin/activate

python3 perf.py -m 1024 -s 0 -e 1024 -i 16 -F matrix-1024
