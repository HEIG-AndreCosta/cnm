
# Lab03 - Session 02 - Neural Networks and OpenMP

Follow the steps of the laboratory and complete the information

## Stage 2

What would happen if all the weights were initialized to 0 instead?

Answer: Le reseaux de neurone apprendra toujours la même chose car les minimus seront tous les memes.

## Stage 3

Provide examples of other activation functions that we could use in this context. 

Answer:
### ReLU (Rectified Linear Unit)
**Définition :**  
ReLU(x) = max(0, x)

**Utilisation :**  
- Couramment utilisée dans les réseaux neuronaux profonds.
- Simple et efficace pour le calcul des gradients.

---

### Tanh (Tangente Hyperbolique)
**Définition :**  
Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**Utilisation :**  
- Produit des sorties centrées autour de 0, ce qui peut accélérer la convergence.

---

### Softmax
**Définition :**  
Softmax(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)

**Utilisation :**  
- Utilisée dans la couche de sortie pour les problèmes de classification.  
- Convertit un vecteur en probabilités normalisées (somme = 1).

---

### Leaky ReLU
**Définition :**  
Leaky ReLU(x) =  
- x, si x > 0  
- αx, sinon (où α est un petit nombre, par ex. 0.01)

**Utilisation :**  
- Version modifiée de ReLU qui atténue le problème des gradients nuls.


## Stage 5 - 6 

Where did you include the pragmas?

Train with/out openMP. Use different number of threads

|          |   time   |
|----------|----------|
| Baseline |0.091317s |
| openMP   |0.034336s | 


## Stage 7

Give an analysis of different places and binds combinations.
