# Lab02 - Session 02 - Convolution and loop unrolling

Follow the steps of the laboratory and complete the information

## Stage 3

Measure convolution time with the images inside /images

| Image                   | Rows | Columns | No Loop Unrolling |
| ----------------------- | ---- | ------- | ----------------- |
| images/bike.jpg         | 480  | 640     | 34598 (+0)        |
| images/bike_edges.png   | 480  | 640     | 34700 (+0)        |
| images/coins.png        | 246  | 300     | 8258 (+0)         |
| images/coins_edges.png  | 246  | 300     | 8254 (+0)         |
| images/engine.png       | 480  | 640     | 34658 (+0)        |
| images/coins_y.png      | 246  | 300     | 8296 (+0)         |
| images/engine_y.png     | 480  | 640     | 34661 (+0)        |
| images/bike_x.png       | 480  | 640     | 34649 (+0)        |
| images/engine_edges.png | 480  | 640     | 34678 (+0)        |
| images/bike_y.png       | 480  | 640     | 34760 (+0)        |
| images/engine_x.png     | 480  | 640     | 34784 (+0)        |
| images/coins_x.png      | 246  | 300     | 8301 (+0)         |

## Stage 4

We are asking you to change the convolution implentation and unroll the loop that iterates over the kernel 3x3.

## Stage 5

En plus de ce qui a été demandé pour cette étape, nous avons aussi comparé la performance de notre implémentation de loop unrolling avec -O0, -O1 et -O2.

En tout, nous avons comparé 7 configurations différentes:

- Pas de loop unrolling
- Loop unrolling manuel avec -O0
- Loop unrolling manuel avec -O1
- Loop unrolling manuel avec -O2
- Loop unrolling (-funroll-loops) avec -O0
- Loop unrolling (-funroll-loops) avec -O1
- Loop unrolling (-funroll-loops) avec -O2

### Compilation

Une fois le fichier Makefile modifié, le lancement de la commande `make` permet de compiler les différentes configurations:

```bash
make -j16
g++ -O0 -c -Wall -I /usr/include/opencv4 -o edge_detection_no_unroll.o edge_detection.cpp
g++ -O0 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING -o edge_detection_manualo0_unroll.o edge_detection.cpp
g++ -O1 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING -o edge_detection_manualo1_unroll.o edge_detection.cpp
g++ -O2 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING -o edge_detection_manualo2_unroll.o edge_detection.cpp
g++ -O0 -c -Wall -I /usr/include/opencv4 -funroll-loops -o edge_detection_compilero0_unroll.o edge_detection.cpp
g++ -O1 -c -Wall -I /usr/include/opencv4 -funroll-loops -o edge_detection_compilero1_unroll.o edge_detection.cpp
g++ -O2 -c -Wall -I /usr/include/opencv4 -funroll-loops -o edge_detection_compilero2_unroll.o edge_detection.cpp
g++ edge_detection_no_unroll.o -o edge_detection_no_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_manualo0_unroll.o -o edge_detection_manualo0_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_compilero0_unroll.o -o edge_detection_compilero0_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_manualo1_unroll.o -o edge_detection_manualo1_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_compilero2_unroll.o -o edge_detection_compilero2_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_compilero1_unroll.o -o edge_detection_compilero1_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_manualo2_unroll.o -o edge_detection_manualo2_unroll -lopencv_imgcodecs -lopencv_core
```

### Résultats

| Image                   | Rows | Columns | No Loop Unrolling | Conv. time (SW loop unrolling) (-O0) | Conv. time (SW loop unrolling) (-O1) | Conv. time (SW loop unrolling) (-O2) | Conv. time (compiler -O0) | Conv. time (compiler -O1) | Conv. time (compiler -O2) |
| ----------------------- | ---- | ------- | ----------------- | ------------------------------------ | ------------------------------------ | ------------------------------------ | ------------------------- | ------------------------- | ------------------------- |
| images/bike.jpg         | 480  | 640     | 34598 (+0)        | 24077 (-10521)                       | 4992 (-29606)                        | 4897 (-29701)                        | 34537 (-61)               | 5013 (-29585)             | 4924 (-29674)             |
| images/bike_edges.png   | 480  | 640     | 34700 (+0)        | 24159 (-10541)                       | 5006 (-29694)                        | 4935 (-29765)                        | 35104 (+404)              | 5033 (-29667)             | 4933 (-29767)             |
| images/coins.png        | 246  | 300     | 8258 (+0)         | 5754 (-2504)                         | 1182 (-7076)                         | 1190 (-7068)                         | 8244 (-14)                | 1219 (-7039)              | 1173 (-7085)              |
| images/coins_edges.png  | 246  | 300     | 8254 (+0)         | 5783 (-2471)                         | 1190 (-7064)                         | 1169 (-7085)                         | 8269 (+15)                | 1211 (-7043)              | 1178 (-7076)              |
| images/engine.png       | 480  | 640     | 34658 (+0)        | 24229 (-10429)                       | 5011 (-29647)                        | 4950 (-29708)                        | 35036 (+378)              | 5062 (-29596)             | 4979 (-29679)             |
| images/coins_y.png      | 246  | 300     | 8296 (+0)         | 5770 (-2526)                         | 1192 (-7104)                         | 1177 (-7119)                         | 8296 (+0)                 | 1202 (-7094)              | 1185 (-7111)              |
| images/engine_y.png     | 480  | 640     | 34661 (+0)        | 24149 (-10512)                       | 5009 (-29652)                        | 4937 (-29724)                        | 34604 (-57)               | 5039 (-29622)             | 4962 (-29699)             |
| images/bike_x.png       | 480  | 640     | 34649 (+0)        | 24322 (-10327)                       | 5105 (-29544)                        | 4974 (-29675)                        | 34759 (+110)              | 5105 (-29544)             | 4939 (-29710)             |
| images/engine_edges.png | 480  | 640     | 34678 (+0)        | 24180 (-10498)                       | 5101 (-29577)                        | 4964 (-29714)                        | 35435 (+757)              | 5121 (-29557)             | 4989 (-29689)             |
| images/bike_y.png       | 480  | 640     | 34760 (+0)        | 24221 (-10539)                       | 5093 (-29667)                        | 5021 (-29739)                        | 34758 (-2)                | 5071 (-29689)             | 5007 (-29753)             |
| images/engine_x.png     | 480  | 640     | 34784 (+0)        | 24250 (-10534)                       | 5134 (-29650)                        | 4994 (-29790)                        | 34704 (-80)               | 5099 (-29685)             | 4992 (-29792)             |
| images/coins_x.png      | 246  | 300     | 8301 (+0)         | 5797 (-2504)                         | 1217 (-7084)                         | 1192 (-7109)                         | 8288 (-13)                | 1218 (-7083)              | 1191 (-7110)              |

### Analyse

On peut voir que avec l'option `-funroll-loops` en `-O0`, on obtient la même performance que la configuration sans loop unrolling. Ceci nous indique le compilateur n'effectue pas de loop unrolling en `-O0`.

En `-O1` et `-O2`, on peut voir que la performance des versions avec loop unrolling manuel est la même que celle avec loop unrolling du compilateur. Ceci nous indique que l'optimisation effectué par le compilateur concernant les loops est la même que celle que nous avons effectué manuellement.

En `-O0`, cependant, nous gagnons beaucoup de performance en effectuant le loop unrolling manuellement. Ceci est dû au fait que le compilateur n'effectue pas de loop unrolling en `-O0`.

En général, on peut voir que le loop unrolling permet d'augmenter la performance de notre implémentation de convolution.
