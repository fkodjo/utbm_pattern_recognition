from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import timeit
from Ressources.CCPD_SubDataset import CCPD_SubDataset

##############################################################################
# Variables globales
##############################################################################
batch_size = 32
epochs = 10

# Pour affichage
loss_array = []
epoch_loss_array = []

plt.ion()  # mode interactif

# Variables globales pour la figure unique
fig = None
axes = None

def show_batch(images, bboxes, ncols=3):
    """
    Affiche un batch d'images et leurs bboxes dans une fenêtre unique,
    qui se met à jour à chaque batch.
    """
    global fig, axes  # utiliser la figure/axes existants

    imgs = images.permute(0, 2, 3, 1).cpu().numpy()
    bboxes_np = bboxes.cpu().numpy()
    batch_size = imgs.shape[0]
    H, W = imgs.shape[1], imgs.shape[2]
    nrows = (batch_size + ncols - 1) // ncols

    # Création de la figure unique si elle n'existe pas encore
    if fig is None or axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
        axes = axes.flatten()

    # Mise à jour des axes existants
    for i, ax in enumerate(axes):
        ax.clear()
        if i < batch_size:
            cx, cy, w, h = bboxes_np[i]
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H

            ax.imshow(imgs[i])
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                 edgecolor="red", linewidth=2, fill=False)
            ax.add_patch(rect)
        ax.axis("off")

    plt.tight_layout()
    plt.pause(0.1)  # durée d'affichage pour ce batch
    

#for batch_idx, (images, bboxes) in enumerate(train_loader):
#    print(f"Batch {batch_idx+1}/{len(train_loader)}")
#    show_batch(images, bboxes, ncols=3)

# Récupérer un batch
#images, bboxes = next(iter(train_loader))

# Affichage demandé dans le TP
#show_batch(images, bboxes, ncols=3)

#print(f'Nombre d\'échantillons dans le split d\'entrainement : {train_data}')
#print(f'Nombre d\'échantillons dans le split de validation : {len(test_data)}')

# Redimensionnement et normalisation des donnees pour le training et la validation




# Chargement des donnees
###[ETAPE 0]###
train_data = CCPD_SubDataset("../../CCPD_SubDataset/train", 'train')
test_data = CCPD_SubDataset("../../CCPD_SubDataset/test", 'test')

## Créer un itérable pour chacun de ces splits
train_loader = DataLoader(train_data, batch_size=9, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=9, shuffle=False, num_workers=4)

##############################################################################
# Definition du CNN ETAPE 1
##############################################################################
###[ETAPE 1]###

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=1)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, padding=1)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5, padding=1)
        
        self.pool_avg = nn.AvgPool2d(1)
        
        self.fc1 = nn.Linear(1728, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_out = nn.Linear(512, 4)  # bbox [cx, cy, w, h]
        
    def forward(self, x):
        
        
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        
        in_size = x.size(0)
        x = x.view(in_size, - 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc_out(x)
        
        #x = self.pool_avg(x)
        
        return torch.sigmoid(x)

##############################################################################
# Instanciation du reseau sur le GPU ou CPU
##############################################################################
net = Net()
#net.cuda()  # Transfert du modèle sur le GPU

##############################################################################
# Initialisation de l optimizer ETAPE 6 (init)
##############################################################################
###### Standard gradient decent (learning_rate=0.01 et momentum=0.9)
#optimizer =
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_fct = nn.MSELoss()
loss_array = []

##############################################################################
# Fonction de Training (pour repeter l'apprentissage sur plusieurs epoques)
##############################################################################
def train(epoch):
    net.train()  # Modele en mode training
    running_loss = 0.0
    plt.ion()  # mode interactif

    for batch_idx, (data, target) in enumerate(train_loader):###[ETAPE 2]###   # Boucle sur les donnees d'apprenitssage des batches 
        
        show_batch(data,target,ncols=3)
   
        # data.cuda()  # Transfert des datas sur le GPU

        optimizer.zero_grad()  # Mise a zero necessaire pour la somme des gradients

        ###[ETAPE 3]###  # Alimentation du reseau (passe forward)
        output = net(data)

        ###[ETAPE 4]###  # Utilisation de la negative log likelihood pour calculer la Loss 
        loss = loss_fct(output, target) 

        ###[ETAPE 5]###  # Passe backward du reseau (calcul de la somme des gradients dloss/dinput)
        loss.backward() 

        ###[ETAPE 6]###  # Mise a jours des parametres du modele (update weights)
        optimizer.step()

        # Pour l'affichage d'infos a la fin de l'entrainement
        running_loss+=loss.item()
        loss_array.append(loss.item())

        # Affichage du status courant de l'entrainement
        if (batch_idx % 10 == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

##############################################################################
# Fonction de Test (pour évaluer l'apprentissage au fur et à mesure)
##############################################################################
def test(epoch):
    net.eval()  # Modele en mode test (ici, on ne veut pas mettre a jour les parametres)
    test_loss = 0
    correct = 0


    for batch_idx, (data, target) in enumerate(test_loader):###[ETAPE 2]###  # Boucle sur les donnees de test des batches 

        ###[ETAPE 3]### # Alimentation du reseau (passe forward)
        output  = net(data) 
 
        ###[ETAPE 4]###  # Utilisation de la negative log likelihood pour calculer la Loss
        test_loss += loss_fct(output, target, reduction='sum').item()
  
        # Pour l'affichage d'infos sur la validation
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  

    test_dat_len = len(test_loader.dataset)
    test_loss /= test_dat_len

    # Afficher la precision du test
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, test_dat_len, 100. * correct / test_dat_len))



##############################################################################
# Fonction main
##############################################################################
if __name__ == '__main__':
    for epoch in range(1, epochs):
        start_time = timeit.default_timer()

        train(epoch)
        test(epoch)

        elapsed = timeit.default_timer() - start_time
        print("Epoch time is", elapsed, "s\n")
    
    # Save the training in W_CNN.pt
    torch.save(net.state_dict(), 'W_CNN.pt')
    # Plot 
    # Plot et sauvegarde au lieu d'afficher
    plt.figure()
    plt.plot(loss_array)
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.show()

    # Sauvegarder les poids du reseau entraine


