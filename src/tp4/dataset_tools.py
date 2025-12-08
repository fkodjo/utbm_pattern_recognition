from torch.utils.data import DataLoader
import numpy as np
import torch, os
from torchvision import datasets , transforms, utils
import matplotlib.pyplot as plt
preprocess = transforms.Compose([
            transforms.Resize([30, 30]),
            transforms.ToTensor()
            ])

# Fonction pour afficher les images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize([30, 30]),
        transforms.ToTensor()
        ]),
    "test" : transforms.Compose([
        transforms.Resize([30, 30]),
        transforms.ToTensor()
        ])
    }

# Chargement des donnees
data_dir = "../../CCPD_Dataset_preproc"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir , x),
                                          data_transforms[x]) for x in ["train", "test"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
batch_size =1, shuffle=True , num_workers =4) for x in ["train", "test"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
class_names = image_datasets['train'].classes

train_loader = dataloaders["train"]

# Parcourir les premiers batches
num_batches_to_show = 3

for batch_idx, (data , target) in enumerate(train_loader):
    print("Batch index :", batch_idx)
    print("Shape de data :", data.shape)
    print("Shape de target :", target.shape)
    print("Target :", target)
    # Créer une grille d'images
    grid = utils.make_grid(data)
    imshow(grid)
    if batch_idx + 1 == num_batches_to_show:
        break

# Convertir le tensor en format compatible avec matplotlib
#plt.imshow(grid.permute(1, 2, 0))  # permute pour avoir H x W x C
#plt.title(f"Labels: {target.tolist()}")
#plt.axis('off')
#plt.show()