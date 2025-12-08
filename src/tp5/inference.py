import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Ressources.training import Net

# Charger le modèle
net = Net()
net.load_state_dict(torch.load("./W_CNN.pt", map_location="cpu"))
net.eval()

# Transforms identiques au dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),   # conserve le RGB
])

def predict_bbox(image_path):
    # Charger l'image originale (pour affichage)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # Transformation pour le réseau
    pil_img = Image.fromarray(img)
    x = transform(pil_img).unsqueeze(0)   # [1, 3, 256, 256]

    # Prédiction
    with torch.no_grad():
        output = net(x)[0]  # [4]

    cx, cy, w, h = output.tolist()

    # Convertir en pixels
    x1 = int((cx - w/2) * W)
    y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W)
    y2 = int((cy + h/2) * H)

    return (x1, y1, x2, y2), img


def show_prediction(image, bbox):
    x1, y1, x2, y2 = bbox

    plt.imshow(image)
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), x2-x1, y2-y1,
                      edgecolor='red', linewidth=2, fill=False)
    )
    plt.show(block=True)


# 🔥 Test
bbox, img = predict_bbox("../../CCPD_SubDataset/test/057022270115-90_89-110&488_518&633-527&620_109&638_131&485_549&467-0_0_26_30_29_31_28-137-250.jpg")
print("Box prédite :", bbox)
show_prediction(img, bbox)
