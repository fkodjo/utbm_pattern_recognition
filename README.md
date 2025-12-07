# A2025_VA52_TP_A

# Explication du reseau

# Analyse des dimensions du réseau de neurones convolutif

🔥 **1) Formule générale avec padding et stride**

Pour une convolution :

$$
\text{Sortie} = \frac{(N - K + 2P)}{S} + 1
$$

avec :  
- \(N\) = taille d'entrée  
- \(K\) = taille du filtre  
- \(P\) = padding  
- \(S\) = stride  

---

🧱 **ÉTAPE 1 — Convolution 1**

- Entrée : 30  
- Kernel : 5  
- Padding : 1  
- Stride : 1  

$$
\text{Sortie} = \frac{30 - 5 + 2(1)}{1} + 1 = 28
$$

👉 La sortie est **28×28**  
👉 Et il y a **10 filtres**  

Donc : **10 × 28 × 28**

---

🪣 **ÉTAPE 2 — Max Pooling 2×2, stride 2**

Le pooling divise la taille par 2 :

$$
28 / 2 = 14
$$

👉 Sortie : **10 × 14 × 14**

---

❓ **ÉTAPE 3 — Convolution 2 (kernel ?)**

On doit retrouver **320** après aplatissement :

$$
320 = 20 \times X \times X
$$

Donc : **20 × 4 × 4 = 320**  

Avant le pooling final, on avait :

👉 **20 × 8 × 8** (puisque 8 / 2 = 4)  

Entrée de la convolution 2 : 14 × 14  

On cherche la taille du kernel \(K\) :

$$
\frac{14 - K}{1} + 1 = 8
$$

$$
14 - K + 1 = 8 \quad \Rightarrow \quad K = 7
$$

🎉 **Conclusion : Kernel = 7×7 pour la convolution 2**

---

📦 **ÉTAPE 4 — Max Pooling 2×2**

- Entrée : 8 × 8  
- Pooling 2×2 → 8 / 2 = 4  

👉 Sortie : **20 × 4 × 4**  

Puis flatten :

$$
20 \times 4 \times 4 = 320
$$

---

✅ **RÉCAPITULATIF FINAL**

| Étape | Taille |
|-------|--------|
| Entrée | 30 × 30 |
| Conv1 (5×5, pad=1, stride=1) | 28 × 28, 10 filtres |
| MaxPool 2×2 | 14 × 14, 10 filtres |
| Conv2 (7×7, pad=0, stride=1) | 8 × 8, 20 filtres |
| MaxPool 2×2 | 4 × 4, 20 filtres |
| Flatten | 320 |
