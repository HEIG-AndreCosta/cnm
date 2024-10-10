#/bin/bash

# set env

source /venv/bin/activate

python3 perf.py -s 0 -e 1024 -i 6 -F matrix-1024
