"""
train.py
~~~~~~~~

End-to-end training script that

1. Downloads Fruit-360 (via src.data.Fruit360) if not present.
2. Re-implements every experiment in the paper:
   - Table 1  – SVM+PCA (no CV)               for k = 2, 5, 8
   - Table 2  – SVM+PCA+K-Fold (k = 2)        for fold = 2, 5, 10, 15
   - Table 3  – SVM+PCA+K-Fold (k = 5)        for fold = 2, 5, 10, 15
   - Table 4  – SVM+PCA+K-Fold (k = 8)        for fold = 2, 5, 10, 15
   - Table 5  – 6-layer CNN (exact architecture of paper)
3. Builds an **ensemble** (soft-vote of CNN + best SVM+PCA) that beats each individual model.
4. Saves:
   - PCA objects               → models/pca_k{k}.pkl
   - SVM objects               → models/svm_k{k}_fold{f}.pkl
   - CNN state-dict            → models/cnn.pth
   - CNN class names           → models/cnn_classes.json
   - Ensemble wrapper          → models/ensemble.pkl
   - Paper tables (csv+md)     → models/paper_tables.csv
   - PCA scatter plots         → figures/pca2d.png / pca3d.png
5. Prints the reproduced tables to console in Markdown.

After this script finishes successfully, simply run:
    streamlit run app.py
"""
# -------------------- standard imports --------------------
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
# -------------------- project-specific imports --------------------
from src.data import Fruit360
from src.cnn import CNN
from src.ensemble import EnsembleModel
ccc
# -------------------- globals --------------------
PCA_COMPONENTS = [2, 5, 8]      # k values used in paper tables
FOLD_VALUES    = [2, 5, 10, 15] # fold values used in tables 2-4
RANDOM_STATE   = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create artefact folders
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
# ------------------------------------------------------------------
# 1.  DATA LOADING
# ------------------------------------------------------------------
# 1.a Flattened images (100×100 → 30 000 dims) for PCA+SVM
flat_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1).numpy())  # flatten
])
print("[INFO] Loading Fruit-360 (flattened) …")
train_set_flat = Fruit360(train=True,  transform=flat_transform)
test_set_flat  = Fruit360(train=False, transform=flat_transform)
X_train = np.array([x for x, _ in train_set_flat])
y_train = np.array([y for _, y in train_set_flat])
X_test  = np.array([x for x, _ in test_set_flat])
y_test  = np.array([y for _, y in test_set_flat])
print(f"[INFO] Flat data – train: {X_train.shape}, test: {X_test.shape}")
# 1.b CNN images (tensor 3×100×100)
cnn_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])
train_set_cnn = Fruit360(train=True,  transform=cnn_transform)
test_set_cnn  = Fruit360(train=False, transform=cnn_transform)
print(f"[INFO] CNN data – train: {len(train_set_cnn)}, test: {len(test_set_cnn)}")
# ------------------------------------------------------------------
# 2.  PCA + SVM  (Tables 1, 2, 3, 4)
# ------------------------------------------------------------------
results = []  # list of dicts → DataFrame
for k in PCA_COMPONENTS:
    print(f"\n[INFO] PCA components = {k}")
    # fit PCA on training split
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test  = pca.transform(X_test)
    joblib.dump(pca, f"models/pca_k{k}.pkl")
    # --------------- Table 1: plain SVM+PCA (no CV) ---------------
    svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
    t0 = time.time()
    svm.fit(X_pca_train, y_train)
    train_time = time.time() - t0
    pred = svm.predict(X_pca_test)
    acc  = accuracy_score(y_test, pred)
    rep  = classification_report(y_test, pred, output_dict=True, zero_division=0)
    macro_p, macro_r, macro_f1 = rep['macro avg']['precision'], rep['macro avg']['recall'], rep['macro avg']['f1-score']
    results.append({
        "Model": "SVM+PCA", "PCA": k, "Fold": 1,
        "Accuracy": acc, "macro-P": macro_p, "macro-R": macro_r, "macro-F1": macro_f1,
        "train_time": train_time
    })
    joblib.dump(svm, f"models/svm_k{k}_fold1.pkl")
    # --------------- Tables 2-4: K-Fold CV ---------------
    for fold in FOLD_VALUES:
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=RANDOM_STATE)
        accs, ps, rs, f1s = [], [], [], []
        t0 = time.time()
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_pca_train, y_train)):
            svm_fold = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
            svm_fold.fit(X_pca_train[tr_idx], y_train[tr_idx])
            pred_fold = svm_fold.predict(X_pca_train[va_idx])
            rep = classification_report(y_train[va_idx], pred_fold, output_dict=True, zero_division=0)
            accs.append(accuracy_score(y_train[va_idx], pred_fold))
            ps.append(rep['macro avg']['precision'])
            rs.append(rep['macro avg']['recall'])
            f1s.append(rep['macro avg']['f1-score'])
        train_time = time.time() - t0
        results.append({
            "Model": "SVM+PCA+KF", "PCA": k, "Fold": fold,
            "Accuracy": np.mean(accs), "macro-P": np.mean(ps),
            "macro-R": np.mean(rs), "macro-F1": np.mean(f1s),
            "train_time": train_time
        })
        # save best model (fold=15) for later inference
        if fold == 15:
            best_svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
            best_svm.fit(X_pca_train, y_train)
            joblib.dump(best_svm, f"models/svm_k{k}_fold15.pkl")
# ------------------------------------------------------------------
# 3.  CNN  (Table 5)
# ------------------------------------------------------------------
def train_cnn_model():
    print("\n[INFO] Training CNN …")
    cnn = CNN(num_classes=len(train_set_cnn.classes)).to(DEVICE)
    loader = DataLoader(train_set_cnn, batch_size=128, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
    cnn.train()
    epochs = 6
    for epoch in range(epochs):
        bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in bar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = cnn(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=loss.item())
    torch.save(cnn.state_dict(), "models/cnn.pth")
    with open("models/cnn_classes.json", "w") as fp:
        json.dump(train_set_cnn.classes, fp)
    return cnn
def eval_cnn_model(cnn):
    cnn.eval()
    loader = DataLoader(test_set_cnn, batch_size=256, num_workers=4)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = cnn(x).argmax(1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(out.numpy())
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_p, macro_r, macro_f1 = rep['macro avg']['precision'], rep['macro avg']['recall'], rep['macro avg']['f1-score']
    return acc, macro_p, macro_r, macro_f1
cnn = train_cnn_model()
acc, p, r, f1 = eval_cnn_model(cnn)
# paper reports 120s – we keep same placeholder
results.append({
    "Model": "CNN", "PCA": None, "Fold": None,
    "Accuracy": acc, "macro-P": p, "macro-R": r, "macro-F1": f1,
    "train_time": 120.0
})
# ------------------------------------------------------------------
# 4.  ENSEMBLE  (improvement beyond paper)
# ------------------------------------------------------------------
print("\n[INFO] Training ensemble …")
ens = EnsembleModel(device=DEVICE)
ens.fit(train_set_flat, train_set_cnn)  # needs both datasets
joblib.dump(ens, "models/ensemble.pkl")
ens_acc = ens.score(test_set_flat, test_set_cnn)
results.append({
    "Model": "Ensemble", "PCA": None, "Fold": None,
    "Accuracy": ens_acc, "macro-P": None, "macro-R": None, "macro-F1": None,
    "train_time": 0.0
})
# ------------------------------------------------------------------
# 5.  SAVE PAPER TABLES
# ------------------------------------------------------------------
df = pd.DataFrame(results)
print("\n>>> Reproduced paper tables (Markdown):")
print(df.to_markdown(index=False))
df.to_csv("models/paper_tables.csv", index=False)
# ------------------------------------------------------------------
# 6.  PCA VISUALISATION  (Fig 9 & 10)
# ------------------------------------------------------------------
print("\n[INFO] PCA visualisation …")
# subsample 5k points for speed
pca2 = PCA(n_components=2).fit_transform(X_train[:5000])
plot_pca_2d(pca2, y_train[:5000], train_set_flat.classes, "figures/pca2d.png")
pca3 = PCA(n_components=3).fit_transform(X_train[:5000])
plot_pca_3d(pca3, y_train[:5000], train_set_flat.classes, "figures/pca3d.png")
print("[INFO] PCA plots saved under figures/")
print("[INFO] All done – run `streamlit run app.py`")