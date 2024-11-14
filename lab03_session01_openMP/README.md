# Lab03 - Session 01 - Multithreading

Follow the steps of the laboratory and complete the information

## Stage 2

What version of OpenMP is implmented in the GNU Compiler Collection available in our boards?

Answer: In order to check for the openmp version we can run the following command.
        Source: https://stackoverflow.com/a/13612520
```bash
        echo |cpp -fopenmp -dM |grep -i open
```

What do we need to do to enable OpenMP?

Answer: In order to compile our code using OpenMP we need to add the flag `-fopenmp` 

### Example 1

Explain the behaviour of the value of the variables during the execution when we declare them as

**private** : Création d'une variable non initialisée qui shadow la variable portant le même nom dans le scope parent. Cela veut dire que les variable que l'on va afficher dans le block ont des valeurs indefinies et que les modifications faites ne modifient pas la variable du scope parent.

exemple:

```bash
cnm@cnm-desktop:~/cnm/lab03_session01_openMP$ ./priv_stage2 
Thread 01 of 06 - Vars -544667396, 43690
Thread 00 of 06 - Vars 1166445824, -1715367752
Thread 04 of 06 - Vars -544667396, 43690
Thread 05 of 06 - Vars -544667396, 43690
Thread 02 of 06 - Vars -544667396, 43690
Thread 03 of 06 - Vars -544667396, 43690
Vars 1,2
```

**firstprivate** : Création d'une variable qui shadow la variable portant le même nom dans le scope parent et qui est initialisée avec la valeur de la variable du scope parent. Cela veut dire que les variable que l'on va afficher dans le block ont des valeurs définies et que les modifications faites ne modifient pas la variable du scope parent.

exemple:

```bash
cnm@cnm-desktop:~/cnm/lab03_session01_openMP$ ./firstpriv_stage2 
Thread 04 of 06 - Vars 1, 2
Thread 01 of 06 - Vars 1, 2
Thread 05 of 06 - Vars 1, 2
Thread 00 of 06 - Vars 1, 2
Thread 02 of 06 - Vars 1, 2
Thread 03 of 06 - Vars 1, 2
Vars 1,2
```

**shared** : Utilisation de la variable du scope parent.
Cela veut dire que les variable que l'on va afficher dans le block ont les mêmes valeurs que celles du scope parent et que les modifications faites modifient la variable du scope parent.

exemple:

```bash
cnm@cnm-desktop:~/cnm/lab03_session01_openMP$ ./shared_stage2
Thread 01 of 06 - Vars 1, 2
Thread 04 of 06 - Vars 2, 3
Thread 02 of 06 - Vars 2, 3
Thread 03 of 06 - Vars 3, 4
Thread 05 of 06 - Vars 4, 5
Thread 00 of 06 - Vars 5, 6
Vars 6,7
```


### Example 2

How much the sequential implementation takes?
```bash 
cnm@cnm-desktop:~/cnm/lab03_session01_openMP$ gcc -fopenmp -o s2p2 stage2_part2.c
cnm@cnm-desktop:~/cnm/lab03_session01_openMP$ ./s2p2_no_parallel 
Sum  400000000 (0.49063s)
```


How much the parallel implementation with OpenMP takes?
```bash
cnm@cnm-desktop:~/cnm/lab03_session01_openMP$ gcc -fopenmp -o s2p2 stage2_part2.c 
cnm@cnm-desktop:~/cnm/lab03_session01_openMP$ ./s2p2 
Sum  400000000 (0.08241s)
```


How have you implimented the parallel dot product with OpenMP?
Does your parallel implementation produce the correct result?
If it does, explain anything you had to consider.
If it does not, explain why.

## Stage 3 Naive implementation matrix

Nous avons lancé plusieur fois le programme avec de differentes valeurs de `k` et le meilleur resultat est obtenu avec `k=50` pour un temps de 

```bash

Prediction time: 0.095306s

```
|   |  Be |  Ma |
|---|-----|-----|
|Be | 122 |   8 |
|Ma |   3 |  36 |


## Stage 5

Measure openMP prediction time

Questions:

* What part(s) of the algorithm have you paralelized?
- kargmin n'a pas pus être optimisé. En effet, paralelliser la boucle exterieur n'a aucune utilité, la boucle interieur peut être paralellisée avec reduction(min:cur_min_distance) mais on a un soucis sur l'assigantion de dist_idx. Il nous faudrait une sorte de if qui check quel thread à trouvé la valeur minimum et assigne dist_idx en conséquence.

* Is it faster than the naive implementation? (explain why or why not)

### Multiple threaded confusion matrix


|    | Be | Ma |
|----|----|----|
| Be |    |    |
| Ma |    |    |
