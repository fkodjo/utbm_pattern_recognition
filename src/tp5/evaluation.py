import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from Ressources.training import Net
from Ressources.CCPD_SubDataset import CCPD_SubDataset

# -------------------------------
# Paramètres
# -------------------------------
test_path = "../../CCPD_SubDataset/test"
model_path = "./W_CNN.pt"
batch_size = 10

# -------------------------------
# Charger le modèle entraîné
# -------------------------------
net = Net()
net.load_state_dict(torch.load(model_path))
net.eval()  # mode évaluation

# -------------------------------
# Charger le dataset de test
# -------------------------------
test_data = CCPD_SubDataset(test_path, split='test')
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

# -------------------------------
# Fonctions utilitaires
# -------------------------------
def compute_iou(boxA, boxB):
    xA1 = boxA[0] - boxA[2]/2
    yA1 = boxA[1] - boxA[3]/2
    xA2 = boxA[0] + boxA[2]/2
    yA2 = boxA[1] + boxA[3]/2

    xB1 = boxB[0] - boxB[2]/2
    yB1 = boxB[1] - boxB[3]/2
    xB2 = boxB[0] + boxB[2]/2
    yB2 = boxB[1] + boxB[3]/2

    xI1 = max(xA1, xB1)
    yI1 = max(yA1, yB1)
    xI2 = min(xA2, xB2)
    yI2 = min(yA2, yB2)

    inter_w = max(0, xI2 - xI1)
    inter_h = max(0, yI2 - yI1)
    inter_area = inter_w * inter_h

    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union_area = areaA + areaB - inter_area

    return inter_area / union_area if union_area > 0 else 0

def show_prediction(img_tensor, bbox_pred, bbox_gt):
    img = img_tensor.permute(1,2,0).cpu().numpy()
    h, w, _ = img.shape
    cx, cy, bw, bh = bbox_pred
    x1 = int((cx - bw/2)*w)
    y1 = int((cy - bh/2)*h)
    x2 = int((cx + bw/2)*w)
    y2 = int((cy + bh/2)*h)
    gcx, gcy, gbw, gbh = bbox_gt
    gx1 = int((gcx - gbw/2)*w)
    gy1 = int((gcy - gbh/2)*h)
    gx2 = int((gcx + gbw/2)*w)
    gy2 = int((gcy + gbh/2)*h)

    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', linewidth=2, fill=False))
    plt.gca().add_patch(plt.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1, edgecolor='green', linewidth=2, fill=False))
    plt.axis('off')
    plt.show(block=True)

# -------------------------------
# Évaluation sur le dataset de test
# -------------------------------
ious = []
losses = []

mse_loss = torch.nn.MSELoss()

for batch_idx, (images, targets) in enumerate(test_loader):
    with torch.no_grad():
        outputs = net(images)

    # Calculer la loss MSE pour ce batch
    batch_loss = mse_loss(outputs, targets)
    losses.append(batch_loss.item())

    for i in range(images.size(0)):
        pred_box = outputs[i].cpu().numpy()
        gt_box = targets[i].cpu().numpy()
        iou = compute_iou(pred_box, gt_box)
        ious.append(iou)

        # Affichage des premières images seulement
        if batch_idx == 0 and i < 5:
            show_prediction(images[i], pred_box, gt_box)

# -------------------------------
# Résultats
# -------------------------------
print(f"IoU moyen sur le dataset de test : {np.mean(ious):.4f}")
print(f"Loss MSE moyenne sur le dataset de test : {np.mean(losses):.6f}")

# -------------------------------
# Tracer les courbes
# -------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(losses, label='Loss MSE')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss par batch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(ious, label='IoU', color='orange')
plt.xlabel('Image index')
plt.ylabel('IoU')
plt.title('IoU par image')
plt.legend()

plt.tight_layout()
plt.show(block=True)
