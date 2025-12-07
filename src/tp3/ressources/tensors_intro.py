from __future__ import print_function
import torch
import numpy as np

########################################################################
# Doc : https://pytorch.org/docs/stable/torch.html#tensors
########################################################################


### Declarations

# Declarer (sans initialiser) un Tensor pyTorch (3x2)
x = torch.empty(3,2)
print(x)

# Declarer et initialiser un Tensor pyTorch (3x2)
x = torch.tensor([[1,2],[4,3],[4,2]])
print(x)

# Declarer un Tensor (4,2) dont tous les elements sont initialise a zero
x = torch.zeros(4,2)
print(x)

# Declarer un Tensor (5x3) aleatoirement initialise
x = torch.rand(5,3)
print(x)

# Declarer un Tensor (3x3) dont les valeurs sont initialisees entre 0 et 1 suivant une loi normale
x = torch.randn(3,3)
print(x)

# Declarer un batch de 10 Tensor (3x3) initilise aleatoirement
batch = torch.rand(10,3,3)
print(batch)

### Operations

# Additionner un Tensor (2x2) avec un scalaire
z = torch.tensor([[2,3],[5,4]]) + 5
print(z)

# Additionner deux Tensors (2x2)
x = torch.tensor([[3,2],[4,3]])
y = torch.tensor([[3,9],[1,3]])
z = x + y
print(x)
print(y)
print(z)

# Mutliplier un Tensor (5x2) par un Tensor (2x3)
a = torch.rand(5,2)
b = torch.rand(2,3)
c = torch.matmul(a,b)
print(a)
print(b)
print(c)

# Mutliplier 2 batch de 3 Tensor (2x2)
batch1 = torch.rand(3,2,2)
batch2 = torch.rand(3,2,2)
r = torch.matmul(batch1,batch2)
print(batch1)
print(batch2)
print(r)
print(batch1[0])
print(batch2[0])
print(r[0])



### Slincing Indexing Joining Mutating

# Declarer un Tensor de 9 elements puis le formater en Tensor (3x3)
v = torch.arange(1, 10)
v = v.view(3,3)
print(v)

# Concatener deux Tensor (2x2)
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.cat((a, b), dim=0)  # concaténation vertical (dim=0)
# c = torch.cat((a, b), dim=1) # concaténation horizontal (dim=1)
print(c)

# Declarer un Tensor (10,5) et afficher sa premiere colonne
a = torch.rand(10, 5)
print(a)
print("First column = ", a[:, 0])


