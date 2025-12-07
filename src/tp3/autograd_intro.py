import torch

#•Déclarer et initialiser un Tensor x de taille 2 ×2
x = torch.ones((2, 2), requires_grad=True)

#•Déclarer et initialiser un Tensor tel que y = x + 5
y = x + 5

#•Déclarer et initialiser un Tensor tel que z = (3 ×y2)/x
z = (3 * y**2) / x

#•Déclarer et initialiser un Tensor tel que o = mean(z)
o = torch.mean(z)
#•Calculer la dérivée ∂o/∂x à la main
#•Vérifier votre calcul de ∂o/∂x à l’aide d’autograd
o.backward()


print(x.grad)