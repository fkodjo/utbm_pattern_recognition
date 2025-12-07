import numpy as np
import os


results = np.load("./p2_results.npy", allow_pickle=True)
TRAIN_DATASET = '../CCPD_Dataset_preproc/train'
TEST_DATASET = '../CCPD_Dataset_preproc/test'

# Toutes les classes connues
classes = sorted(os.listdir(TRAIN_DATASET))

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

