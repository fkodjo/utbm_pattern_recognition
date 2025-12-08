import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timeit
from Ressources.training import Net  # Importer la définition du modèle depuis training.py
import os
# Charger le modèle entraîné
model_path = "./W_CNN.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier du modèle entraîné '{model_path}' est introuvable.")
# Charger le modèle
net = Net()
net.load_state_dict(torch.load(model_path))
net.eval()  # Mettre le modèle en mode évaluation

# Définir les transformations pour les images d'entrée
transform = transforms.Compose([
    transforms.Resize((30, 30)),  # Redimensionner l'image à 30x30 pixels
    transforms.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris
    transforms.ToTensor(),  # Convertir en tenseur PyTorch
])
def predict_image(image_path):
    # Charger l'image
    image = Image.open(image_path)
    # Appliquer les transformations
    image = transform(image)
    # Ajouter une dimension batch
    image = image.unsqueeze(0)  # Shape: [1, 1, 30, 30]
    # Faire la prédiction
    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Tester le modèle sur une image d'entrée
test_image_path = "../../CCPD_Dataset_preproc/test/4/40.png"  # Remplacer par le chemin de votre image
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"L'image de test '{test_image_path}' est introuvable.")
predicted_class = predict_image(test_image_path)
print(f"La classe prédite pour l'image '{test_image_path}' est : {predicted_class}")