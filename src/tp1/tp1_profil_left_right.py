import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from collections import defaultdict

TRAIN_DATASET = '../CCPD_Dataset_preproc/train'
TEST_DATASET = '../CCPD_Dataset_preproc/test'
n_lines = 5

class ProfilLeftRight:
    
    def __init__(self, image_grayscale):
        self.image_grayscale = image_grayscale
    
    def extract_profils(self, x_min, y_min, x_max, y_max, n_lignes=5):
        char = self.image_grayscale[y_min:y_max,x_min:x_max]
        
        # Binary image
        _, binary = cv.threshold(self.image_grayscale, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        #h, w = binary.shape
        #binary = char.copy()  # ou 255 - char si inversion nécessaire
        h, w = self.image_grayscale.shape
        
        if h == 0 or w == 0:
            print("Image vide ou aucun pixel noir trouvé →", self.image_grayscale)
            return np.zeros(n_lignes*2)
        
        # Ligne uniformement reparti
        ys = np.linspace(0, h - 1, n_lignes).astype(int)
        
        profil_left = []
        profil_right = []
        
        for y in ys:
            row = binary[y]
            if row.size == 0:
                left = w
                right = w
            else:
                left = np.argmax(row > 0)
                right = np.argmax(row[::-1] > 0)
                if np.max(row) == 0:
                    left = w
                    right = w
            
                # Profil left
                #left = np.argmax(row > 0)
                #if row.max() == 0:
                #    left = w
                    
                # Profil right
                #right = np.argmax(row[::-1] > 0)
                #if row.max() == 0:
                #    right = w
            
            # Normalisation par la largeur
            profil_left.append(left/w)
            
            profil_right.append(right/w)
            
        # 4. Construire le vecteur final (gauche + droite)
        vector = np.array(profil_left + profil_right)
        
        return vector

def train():
    centers = {}
    for classe in os.listdir(TRAIN_DATASET):
        class_path = os.path.join(TRAIN_DATASET, classe)
        if not os.path.isdir(class_path):
            continue
        
        vectors = []
        
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            # bounding box du caractère (selon ton dataset)
            # pixels noirs => caractère
            y, x = np.where(img < 255)
            if len(x) == 0 or len(y) == 0:
                print("ERREUR : image vide ou aucun pixel noir trouvé →", img_path)
                continue
            
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            
            extractor = ProfilLeftRight(img)
            vector = extractor.extract_profils(x_min, y_min, x_max, y_max, n_lines)
            
            vectors.append(vector)
        
        vectors = np.array(vectors)
        centre = np.mean(vectors, axis=0)
        centers[classe] = centre
        
    np.save("centers_profils.npy", centers)
    print("Centres des classes sauvegardés dans centres_profils.npy /n", centers)
    

def test():
    # Charger les centres
    centres = np.load('./centers_profils.npy', allow_pickle=True).item()

    resultats = {}  # pour stocker les probabilités

    for classe in os.listdir(TEST_DATASET):
        class_path = os.path.join(TEST_DATASET, classe)
        if not os.path.isdir(class_path):
            continue

        resultats[classe] = []

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            
            img = img.astype(np.uint8)

            # bounding box du caractère
            y, x = np.where(img < 255)
            if len(x) == 0 or len(y) == 0:
                print("ERREUR : image vide ou aucun pixel noir trouvé →", img_path)
                continue
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            
            # Forcer une taille minimale
            if (x_max - x_min) < 2: x_max = x_min + 2
            if (y_max - y_min) < 2: y_max = y_min + 2

            extractor = ProfilLeftRight(img)
            x_vec = extractor.extract_profils(x_min, y_min, x_max, y_max, n_lines)

            # --- Étape 1 : distances aux centres ---
            distances = []
            for classe_centre, w in centres.items():
                d = np.linalg.norm(x_vec - w)
                distances.append(d)

            distances = np.array(distances)

            # --- Étape 2 : transformation en probabilités ---
            numerateur = np.exp(-distances)
            proba = numerateur / np.sum(numerateur)

            # sauvegarde
            resultats[classe].append({
                "image": filename,
                "proba": proba.tolist()
            })

    # Sauvegarde finale pour TP2
    np.save("probabilites_profils.npy", resultats)
    print("Probabilités sauvegardées dans probabilites_profils.npy")
    
train()
test()

# Charger les résultats
res = np.load("probabilites_profils.npy", allow_pickle=True).item()

# Toutes les classes connues (dans l’ordre alphabetique)
classes = sorted(res.keys())
classe_to_index = {c: i for i, c in enumerate(classes)}

# Compteurs
correct_total = 0
total = 0

correct_per_class = {c: 0 for c in classes}
total_per_class = {c: 0 for c in classes}

for true_class, entries in res.items():
    for e in entries:
        proba = np.array(e["proba"])
        
        # Classe prédite = argmax
        pred_idx = np.argmax(proba)
        pred_class = classes[pred_idx]
        
        # Mise à jour des compteurs
        total += 1
        total_per_class[true_class] += 1
        
        if pred_class == true_class:
            correct_total += 1
            correct_per_class[true_class] += 1

# Taux global
taux_global = correct_total / total

print("\n===== TAUX DE RECONNAISSANCE =====")
print(f"Taux global : {taux_global*100:.2f}%\n")

print("Taux par classe :")
for c in classes:
    if total_per_class[c] > 0:
        taux = correct_per_class[c] / total_per_class[c]
        print(f"{c} : {taux*100:.2f}% ({correct_per_class[c]}/{total_per_class[c]})")