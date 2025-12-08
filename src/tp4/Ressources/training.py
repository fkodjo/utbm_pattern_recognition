from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import timeit

##############################################################################
# Variables globales
##############################################################################
batch_size = 32
epochs = 10

# Pour affichage
loss_array = []
epoch_loss_array = []

# Redimensionnement et normalisation des donnees pour le training et la validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([30, 30]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize([30, 30]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]),
}

# Chargement des donnees
###[ETAPE 0]###
data_dir = "../../CCPD_Dataset_preproc"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir , x),
                                          data_transforms[x]) for x in ["train", "test"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size = batch_size,
                                              shuffle=True ,
                                              num_workers =4) for x in ["train", "test"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
class_names = image_datasets['train'].classes
train_loader = dataloaders['train']

##############################################################################
# Definition du CNN ETAPE 1
##############################################################################
###[ETAPE 1]###

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1_fct = nn.Conv2d(in_channels=1, out_channels=10,kernel_size=5, stride=1, padding=1)
        self.conv2_fct = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=7, stride=1, padding=0)
        self.mp2 = nn.MaxPool2d(2, padding=0)
        self.drop2D = nn.Dropout2d(p=0.5, inplace=False)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 34)
        
    def forward(self, x):
        in_size = x.size(0)
        
        x = self.conv1_fct(x)
        x = self.mp2(x)
        x = F.relu(x)
        
        x = self.conv2_fct(x)
        x = self.mp2(x)
        x = F.relu(x)
        
        x = self.drop2D(x)
        x = x.view(in_size, - 1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


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
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

##############################################################################
# Fonction de Training (pour repeter l'apprentissage sur plusieurs epoques)
##############################################################################
def train(epoch):
    net.train()  # Modele en mode training
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloaders['train']):###[ETAPE 2]###   # Boucle sur les donnees d'apprenitssage des batches 
   
        # data.cuda()  # Transfert des datas sur le GPU

        optimizer.zero_grad()  # Mise a zero necessaire pour la somme des gradients

        ###[ETAPE 3]###  # Alimentation du reseau (passe forward)
        output = net(data)

        ###[ETAPE 4]###  # Utilisation de la negative log likelihood pour calculer la Loss 
        loss = F.nll_loss(output, target) 

        ###[ETAPE 5]###  # Passe backward du reseau (calcul de la somme des gradients dloss/dinput)
        loss.backward() 

        ###[ETAPE 6]###  # Mise a jours des parametres du modele (update weights)
        optimizer.step()

        # Pour l'affichage d'infos a la fin de l'entrainement
        running_loss+=loss.item()
        loss_array.append(loss.item())

        # Affichage du status courant de l'entrainement
        if (batch_idx % 1 == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(dataloaders['train'].dataset), 100. * batch_idx / len(dataloaders['train']), loss.item()))

##############################################################################
# Fonction de Test (pour évaluer l'apprentissage au fur et à mesure)
##############################################################################
def test(epoch):
    net.eval()  # Modele en mode test (ici, on ne veut pas mettre a jour les parametres)
    test_loss = 0
    correct = 0


    for batch_idx, (data, target) in enumerate(dataloaders['test']):###[ETAPE 2]###  # Boucle sur les donnees de test des batches 

        ###[ETAPE 3]### # Alimentation du reseau (passe forward)
        output  = net(data) 
 
        ###[ETAPE 4]###  # Utilisation de la negative log likelihood pour calculer la Loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
  
        # Pour l'affichage d'infos sur la validation
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  

    test_dat_len = len(dataloaders['test'].dataset)
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


