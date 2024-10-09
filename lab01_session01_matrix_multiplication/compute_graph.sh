#/bin/bash

# set env

source /venv/bin/activate

python3 perf.py -s 10 -e 1000 -i 10 -F naive10-1000
python3 perf.py -s 10 -e 1000 -i 10 -t 2 -F tile10-1000-2
python3 perf.py -s 10 -e 1000 -i 10 -t 4 -F tile10-1000-4
python3 perf.py -s 10 -e 1000 -i 10 -t 10 -F tile10-1000-10
python3 perf.py -s 10 -e 1000 -i 10 -t 20 -F tile10-1000-20
python3 perf.py -s 10 -e 1000 -i 10 -t 50 -F tile10-1000-50
python3 perf.py -s 10 -e 1000 -i 10 -t 100 -F tile10-1000-100
python3 perf.py -s 10 -e 1000 -i 10 -t 200 -F tile10-1000-200