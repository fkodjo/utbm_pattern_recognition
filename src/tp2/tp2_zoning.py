import numpy as np
import cv2 as cv
import os

TRAIN_DATASET = '../CCPD_Dataset_preproc/train'
TEST_DATASET = '../CCPD_Dataset_preproc/test'

class Zoning:
    
    def __init__(self, image):
        self.image_grayscale = image
        
    def extract_zoning_features(self, n=5, m=5):
        """
        Extrait les densités de pixels par zonage pour un caractère.
        
        Args:
            char_img: image du caractère (grayscale ou binaire)
            n: nombre de zones verticales
            m: nombre de zones horizontales
            
        Returns:
            vecteur 1D de densité par zone, normalisé
        """
        # Assurer que l'image est binaire (0 ou 1)
        _, binary = cv.threshold(self.image_grayscale, 127, 1, cv.THRESH_BINARY)
        
        h, w = binary.shape
        feature_vector = []

        zone_h = h // n
        zone_w = w // m

        for i in range(n):
            for j in range(m):
                # Extraire la zone
                zone = binary[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
                # Densité = nombre de pixels "actifs" / surface de la zone
                density = np.sum(zone) / zone.size if zone.size > 0 else 0
                feature_vector.append(density)

        return np.array(feature_vector)
    

def train():
    # Préparer les listes pour stocker les caractéristiques et labels
    X_train = []
    y_train = []

    # Parcourir chaque classe (dossier)
    for classe in os.listdir(TRAIN_DATASET):
        classe_path = os.path.join(TRAIN_DATASET, classe)
        if not os.path.isdir(classe_path):
            continue
        # Parcourir chaque image de la classe
        for filename in os.listdir(classe_path):
            filepath = os.path.join(classe_path, filename)
            img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            extract_zoning = Zoning(img)
            features = extract_zoning.extract_zoning_features(n=5, m=5)
            X_train.append(features)
            y_train.append(classe)

    # Convertir en numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Sauvegarder les vecteurs et labels dans un fichier
    np.savez('train_data.npz', X=X_train, y=y_train)

    print("Apprentissage terminé, vecteurs sauvegardés.")
    

K = 5  # nombre de voisins
CLASSES = sorted(os.listdir(TRAIN_DATASET))  # classes disponibles
N_CLASSES = len(CLASSES)

def knn_probabilities(x, X_train, y_train, K):
    # Calcul des distances euclidiennes
    distances = np.linalg.norm(X_train - x, axis=1)

    # Indices des K plus proches voisins
    idx = np.argsort(distances)[:K]

    # Comptage par classe
    k_counts = np.zeros(N_CLASSES)

    for i in idx:
        class_label = y_train[i]
        class_index = CLASSES.index(class_label)
        k_counts[class_index] += 1

    # Probabilités : k_i / K
    return k_counts / K


def test():
    data = np.load('train_data.npz')
    X_train = data['X']
    y_train = data['y']

    p2_list = []   # stocker les vecteurs de probas
    results = []

    TEST_DATASET = '../CCPD_Dataset_preproc/test'

    for classe in os.listdir(TEST_DATASET):
        classe_path = os.path.join(TEST_DATASET, classe)
        if not os.path.isdir(classe_path):
            continue

        for filename in os.listdir(classe_path):
            filepath = os.path.join(classe_path, filename)
            img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Extraire les caractéristiques
            features = Zoning(img).extract_zoning_features()

            # Calculer le vecteur p₂
            p2 = knn_probabilities(features, X_train, y_train, K)

            # Sauvegarder
            p2_list.append(p2)
            
            results.append({
                "true_class": classe,
                "proba": p2
            })

    p2_array = np.array(p2_list)
    np.save('p2_test.npy', p2_array)

    print("Probabilités p₂ sauvegardées dans p2_test.npy.")

    # Sauvegarde
    np.save("./p2_results.npy", results)
    print("Résultats sauvegardés dans p2_results.npy")

    
#train()
#test()

results = np.load("./p2_results.npy", allow_pickle=True)

# Toutes les classes connues
classes = CLASSES

correct_total = 0
total = 0

correct_per_class = {c: 0 for c in classes}
total_per_class = {c: 0 for c in classes}

for r in results:
    true_class = r["true_class"]
    proba = np.array(r["proba"])

    pred_idx = np.argmax(proba)
    pred_class = classes[pred_idx]

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
        print(f"{c} : {taux*100:.2f}% "
              f"({correct_per_class[c]}/{total_per_class[c]})")

