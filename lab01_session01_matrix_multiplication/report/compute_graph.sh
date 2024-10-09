#/bin/bash

# Duration measurement

python3 perf.py --start 10 -e 1000 -i 10 -T -S -F naive10-1000.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 2 -T -S -F tile10-1000-2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 4 -T -S -F tile10-1000-4.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 10 -T -S -F tile10-1000-10.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 20 -T -S -F tile10-1000-20.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 50 -T -S -F tile10-1000-50.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 100 -T -S -F tile10-1000-100.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 200 -T -S -F tile10-1000-200.svg

# cache L1 measurement
python3 perf.py --start 10 -e 1000 -i 10 -c --cache-type L1 -S -F naive10-1000c.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 2 -c --cache-type L1 -S -F tile10-1000-2cl1.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 4 -c --cache-type L1 -S -F tile10-1000-4cl1.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 10 -c --cache-type L1 -S -F tile10-1000-10cl1.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 20 -c --cache-type L1 -S -F tile10-1000-20cl1.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 50 -c --cache-type L1 -S -F tile10-1000-50cl1.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 100 -c --cache-type L1 -S -F tile10-1000-100cl1.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 200 -c --cache-type L1 -S -F tile10-1000-200cl1.svg

# cache L2 measurement
python3 perf.py --start 10 -e 1000 -i 10 -c --cache-type L2 -S -F naive10-1000cL2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 2 -c --cache-type L2 -S -F tile10-1000-2cl2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 4 -c --cache-type L2 -S -F tile10-1000-4cl2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 10 -c --cache-type L2 -S -F tile10-1000-10cl2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 20 -c --cache-type L2 -S -F tile10-1000-20cl2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 50 -c --cache-type L2 -S -F tile10-1000-50cl2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 100 -c --cache-type L2 -S -F tile10-1000-100cl2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 200 -c --cache-type L2 -S -F tile10-1000-200cl2.svg

# cache both L1 and L2 measurement

python3 perf.py --start 10 -e 1000 -i 10 -c --cache-type both -S -F naive10-1000cl1l2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 2 -c --cache-type both -S -F tile10-1000-2cl1l2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 4 -c --cache-type both -S -F tile10-1000-4cl1l2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 10 -c --cache-type both -S -F tile10-1000-10cl1l2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 20 -c --cache-type both -S -F tile10-1000-20cl1l2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 50 -c --cache-type both -S -F tile10-1000-50cl1l2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 100 -c --cache-type both -S -F tile10-1000-100cl1l2.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 200 -c --cache-type both -S -F tile10-1000-200cl1l2.svg

# full

python3 perf.py --start 10 -e 1000 -i 10 -c -T --cache-type both -S -F naive10-1000cl1l2full.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 2 -c -T --cache-type both -S -F tile10-1000-2cl1l2full.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 4 -c -T --cache-type both -S -F tile10-1000-4cl1l2full.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 10 -c -T --cache-type both -S -F tile10-1000-10cl1l2full.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 20 -c -T --cache-type both -S -F tile10-1000-20cl1l2full.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 50 -c -T --cache-type both -S -F tile10-1000-50cl1l2full.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 100 -c -T --cache-type both -S -F tile10-1000-100cl1l2full.svg
python3 perf.py --start 10 -e 1000 -i 10 -t 200 -c -T --cache-type both -S -F tile10-1000-200cl1l2full.svg






