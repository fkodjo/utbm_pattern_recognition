import torch


# •Créer un block de convolution 1D ayant en entrée 1 canal et 2 en sortie (afficher ses paramètres),
conv_1d = torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1)
print("weight = ", conv_1d.weight, "\n")
print("Bias = ", conv_1d.bias, "\n")

# •Déclarer x, un Tensor 1D d’un seul élément aléatoirement initialisé,
x = torch.randn(1,1,1, requires_grad=True)
print("x = ", x)

# •Alimenter votre couche de convolution avec votre Tensor x et conserver la sortie obtenue dans un Tensor y
y = conv_1d(x)
print("y =", y, "\n")

# •Vérifier manuellement le contenu de y
y_0 = conv_1d.weight[0,0,0] * x + conv_1d.bias[0]
y_1 = conv_1d.weight[1,0,0] * x + conv_1d.bias[1]
print("y_0 = ", y_0, "\n\ny_1 = ",y_1,"\n")

# •Déclarer o, un Tensor 1D initialisé comme  ́etant la somme des  ́eléments de y
o = torch.sum(y)

# •Calculer manuellement ∂o/∂x
o_manuel = y_0 + y_1

print("o manuel = ", o_manuel,"\n")

# •Vérifier votre calcul de ∂o/∂x à l’aide d’autograd
o.backward()
print("Descent Gradient = ",x.grad)