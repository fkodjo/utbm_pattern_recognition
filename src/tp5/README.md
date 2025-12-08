x = self.conv1(x)
x = F.max_pool2d(F.relu(x), 2)
...
x = torch.flatten(x, 1)
print(x.shape)  # affiche [batch_size, flatten_size]



Paramètres de ton CNN

Input : 3 × 256 × 256

Conv1 : 3 → 6, kernel 5, padding 1

Taille : (256 - 5 + 2*1)/1 + 1 = 254

Après MaxPool 2×2 : 254 / 2 = 127 → 6 × 127 × 127

Conv2 : 6 → 12, kernel 5, padding 1

Conv : (127 - 5 + 2*1)/1 + 1 = 125

MaxPool 2×2 : 125 / 2 = 62 (entier tronqué) → 12 × 62 × 62

Conv3 : 12 → 24, kernel 5, padding 1

Conv : (62 - 5 + 2)/1 + 1 = 60

MaxPool 2×2 : 60 / 2 = 30 → 24 × 30 × 30

Conv4 : 24 → 48, kernel 5, padding 1

Conv : (30 - 5 + 2)/1 + 1 = 28

MaxPool 2×2 : 28 / 2 = 14 → 48 × 14 × 14

Conv5 : 48 → 192, kernel 5, padding 1

Conv : (14 - 5 + 2)/1 + 1 = 12

AvgPool 2×2 : 12 / 2 = 6

AvgPool 2×2 encore : 6 / 2 = 3 → 192 × 3 × 3 = 1728

✅ Donc, le flatten avant la première couche fully connected doit être 192 × 3 × 3 = 1728, et non 76800 comme dans ton code actuel.






I. Dimensionnement des couches du CNN

Ton réseau Net est structuré ainsi :

Input: Image RGB 30x30
Conv1: 3 -> 6, kernel 5x5, padding=1
MaxPool: 2x2
Conv2: 6 -> 12, kernel 5x5, padding=1
MaxPool: 2x2
Conv3: 12 -> 24, kernel 5x5, padding=1
MaxPool: 2x2
Conv4: 24 -> 48, kernel 5x5, padding=1
MaxPool: 2x2
Conv5: 48 -> 192, kernel 5x5, padding=1
AvgPool: 2x2
Flatten
FC1: 76800 -> 1024
FC2: 1024 -> 512
FC_out: 512 -> 4

🔹 Comment calculer les tailles intermédiaires

Pour une convolution 2D :

𝑂
=
𝑊
−
𝐾
+
2
𝑃
𝑆
+
1
O=
S
W−K+2P
	​

+1

où :

W = taille d’entrée

K = taille du kernel

P = padding

S = stride

Et pour le pooling 2x2 (stride=2) : on divise la dimension par 2.

Exemple pour Conv1 + MaxPool :

Input : 30x30

Conv1 (kernel=5, padding=1, stride=1) :

𝑂
=
30
−
5
+
2
∗
1
1
+
1
=
28
O=
1
30−5+2∗1
	​

+1=28

MaxPool 2x2 : 28/2 = 14

Channels : 6 → output = (6, 14, 14)

En répétant ce calcul couche par couche, tu peux vérifier que le flatten donne bien 76800 entrées pour FC1.

💡 Astuce : pour éviter les erreurs, imprime les tailles après chaque couche :

print(x.shape)