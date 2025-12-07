# A2025_VA52_TP_A

# Explication du reseau

🔥 1) Formule générale avec padding et stride

Pour une convolution :

Sortie
=
(
𝑁
−
𝐾
+
2
𝑃
)
𝑆
+
1
Sortie=
S
(N−K+2P)
	​

+1

avec :

𝑁
N = taille d'entrée

𝐾
K = taille du filtre

𝑃
P = padding

𝑆
S = stride

🧱 ÉTAPE 1 — Convolution 1

Entrée : 30

Kernel : 5

Padding : 1

Stride : 1

Sortie
=
30
−
5
+
2
(
1
)
1
+
1
Sortie=
1
30−5+2(1)
	​

+1

Calcul :

30
−
5
=
2525
+
2
=
2727
+
1
=
28
30−5=2525+2=2727+1=28

👉 La sortie est 28×28
👉 Et il y a 10 filtres

Donc :

10
×
28
×
28
10×28×28
🪣 ÉTAPE 2 — Max Pooling 2×2, stride 2

Le pooling divise la taille par 2 :

28
/
2
=
14
28/2=14

👉 10 × 14 × 14

❓ ÉTAPE 3 — Convolution 2 (kernel ?)

On doit retrouver 320 après aplatissement :

320
=
20
×
𝑋
×
𝑋
320=20×X×X

Donc :

20
×
4
×
4
=
320
20×4×4=320

Donc après le deuxième pooling, on doit obtenir :

👉 20 × 4 × 4

Ce qui signifie qu’avant ce pooling, on avait :

👉 20 × 8 × 8
(puisque 8 / 2 = 4)

🔍 Donc la sortie de la convolution 2 doit être :

8 × 8

Entrée de la convolution 2 :

👉 14 × 14

On cherche la taille du kernel 
𝐾
K de cette 2ᵉ convolution :

14
−
𝐾
1
+
1
=
8
1
14−K
	​

+1=8

On résout :

14
−
𝐾
+
1
=
8
14−K+1=8
15
−
𝐾
=
8
15−K=8
𝐾
=
7
K=7
🎉 Conclusion : Kernel = 7×7 pour la convolution 2

Ce choix est le seul compatible avec la sortie finale de 320.

📦 ÉTAPE 4 — Max Pooling 2×2

On avait 8 × 8, donc :

8
/
2
=
4
8/2=4

👉 20 × 4 × 4

Puis flatten :

20
×
4
×
4
=
320
20×4×4=320
✅ RÉCAPITULATIF FINAL (simple et clair)
Étape	Taille
Entrée	30×30
Conv1 (5×5, pad=1, stride=1)	28×28, 10 filtres
MaxPool 2×2	14×14, 10 filtres
Conv2 (7×7, pad=0, stride=1)	8×8, 20 filtres
MaxPool 2×2	4×4, 20 filtres
Flatten	320