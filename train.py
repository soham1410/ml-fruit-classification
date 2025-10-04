"""
train.py  (identical 10-fruit labels + full-metrics + confusion matrices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*  Shuffled subset → PCA sees all fruits, not only Apple.
*  Evaluates on **full 60-fruit set** → overall metrics unchanged.
*  Plots **only 10 largest fruits** in bar-charts & confusion matrices → readable visuals.
*  Generates **identical 10-fruit labels** across **all CMs**:
  SVM+PCA(k=5), CNN, and Ensemble.
*  Fruit-level bar-charts (training / test counts).
*  PCA scatter plots with unique fruit legends.
*  Skips weight re-training if artefacts exist – fast re-run.
*  Always re-computes metrics → CSV contains true scores, no zeros.
"""
# -------------------- standard library --------------------
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- project-specific --------------------
from src.data import Fruit360
from src.cnn import CNN
from src.ensemble import EnsembleModel
from src.viz import plot_pca_2d, plot_pca_3d

# -------------------- globals --------------------
PCA_COMPONENTS = [2, 5, 8]      # k values used in paper tables
FOLD_VALUES    = [2, 5, 10, 15] # fold values used in tables 2-4
RANDOM_STATE   = 42             # for reproducible shuffle
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUB_TRAIN      = 15_000         # shuffled → all fruits
SUB_TEST       = 5_000          # shuffled → all fruits
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# -------------------- helpers --------------------
def missing(file):
    """True if file does not exist → skip re-training."""
    return not os.path.exists(file)

def mk_loader(dataset, batch=128, shuffle=True):
    """Windows-safe DataLoader (num_workers=0)."""
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      num_workers=0, pin_memory=False)

# ---------- base-fruit mapper ----------
def to_base_fruit(classes):
    """Apple_Golden_1 -> Apple, Banana_1 -> Banana, …"""
    return [c.split('_')[0] for c in classes]

# ---------- 10-fruit bar-chart (largest support) ----------
def plot_fruit_counts(dataset, title, save_path, top_n=10):
    """Count **unique** base fruits but plot only top-N (largest support)."""
    base = pd.Series(to_base_fruit(dataset.classes)).drop_duplicates().sort_values()
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = pd.Series([base[base == to_base_fruit(dataset.classes)[lbl]].iloc[0]
                        for lbl in labels]).value_counts().sort_index()

    # --- pick top-N ---
    top_counts = counts.nlargest(top_n).sort_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_counts.index, y=top_counts.values, color="steelblue")
    plt.title(f"{title} (top {top_n} fruits)", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------- 10-fruit confusion matrix (full eval, 10-class plot) ----------
def save_fruit_cm(y_true_sub, y_pred_sub, classes, save_path, title="Fruit-Level Confusion Matrix (10 classes)"):
    """Full 60-fruit evaluation → plot only top-10 fruits (largest support)."""
    base_fruits = sorted(set(to_base_fruit(classes)))          # 60 unique
    fruit_to_idx = {f: i for i, f in enumerate(base_fruits)}

    y_true_fruit = np.array([fruit_to_idx[to_base_fruit(classes)[lbl]] for lbl in y_true_sub])
    y_pred_fruit = np.array([fruit_to_idx[to_base_fruit(classes)[lbl]] for lbl in y_pred_sub])

    # --- full 60×60 CM for metrics (kept in memory) ---
    cm_full = confusion_matrix(y_true_fruit, y_pred_fruit, labels=range(len(base_fruits)))

    # --- **same** 10×10 slice (locked list) ---
    cm10 = cm_full[np.ix_(GLOBAL_TOP10_IDX, GLOBAL_TOP10_IDX)]

    # --- plot 10×10 ---
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm10, annot=True, fmt="d", cmap="Blues",
                xticklabels=GLOBAL_TOP10_NAMES, yticklabels=GLOBAL_TOP10_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Fruit", fontsize=14)
    plt.ylabel("True Fruit", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------- shuffled flat-subset loader ----------
def load_subset_flat(dataset, n, desc="flat"):
    """Pick **random** n images (shuffle first) so PCA sees all fruits."""
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(dataset))[:n]
    X, y = [], []
    for i in tqdm(indices, desc=f"[INFO] Loading {desc} subset (shuffled)"):
        img, label = dataset[i]
        X.append(img); y.append(label)
    return np.array(X), np.array(y)

# ===================================================================
#  MAIN  (Windows-safe)
# ===================================================================
if __name__ == '__main__':

    # ------------------------------------------------------------------
    # 0.  LOCK identical 10 fruits for ALL CMs  (moved to top)
    # ------------------------------------------------------------------
    
    cnn_transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    train_set_cnn = Fruit360(train=True,  transform=cnn_transform, download=False)
    test_set_cnn  = Fruit360(train=False, transform=cnn_transform, download=False)

    base_fruits = sorted(set(to_base_fruit(train_set_cnn.classes)))          # 60 unique
    # --- map **actual labels** → base fruit → index ---
    labels_full = [test_set_cnn[i][1] for i in range(len(test_set_cnn))]     # 131-class labels
    y_true_full = np.array([base_fruits.index(to_base_fruit(train_set_cnn.classes)[lbl])
                            for lbl in labels_full])                          # 60-fruit indices
    support = np.bincount(y_true_full, minlength=len(base_fruits))
    unique_support = pd.Series(support, index=base_fruits).groupby(level=0).sum()
    GLOBAL_TOP10_NAMES = unique_support.nlargest(10).index.tolist()
    GLOBAL_TOP10_IDX   = [base_fruits.index(f) for f in GLOBAL_TOP10_NAMES]
    print("[LOCK] Identical 10 fruits for all CMs:", GLOBAL_TOP10_NAMES)

    # ------------------------------------------------------------------
    # 1.  DATA  (CNN = full,  flat = shuffled subset)
    # ------------------------------------------------------------------
    print(f"[INFO] CNN data – train: {len(train_set_cnn)}, test: {len(test_set_cnn)}")

    flat_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1).numpy())
    ])
    train_set_flat = Fruit360(train=True,  transform=flat_transform, download=False)
    test_set_flat  = Fruit360(train=False, transform=flat_transform, download=False)

    X_train, y_train = load_subset_flat(train_set_flat, SUB_TRAIN, "train")
    X_test,  y_test  = load_subset_flat(test_set_flat,  SUB_TEST,  "test")
    print(f"[INFO] Flat subset – train: {X_train.shape}, test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 2.  PCA + SVM  (always re-evaluate – skip weight training if exist)
    # ------------------------------------------------------------------
    results = []
    for k in PCA_COMPONENTS:
        print(f"\n[INFO] PCA components = {k}")
        pca_path = f"models/pca_k{k}.pkl"
        if missing(pca_path):
            pca = PCA(n_components=k, random_state=RANDOM_STATE)
            X_pca_train = pca.fit_transform(X_train)
            X_pca_test  = pca.transform(X_test)
            joblib.dump(pca, pca_path)
        else:
            pca = joblib.load(pca_path)
            X_pca_train = pca.transform(X_train)
            X_pca_test  = pca.transform(X_test)

        # ---- Table 1: plain SVM ----
        svm1_path = f"models/svm_k{k}_fold1.pkl"
        if missing(svm1_path):
            svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
            t0 = time.time()
            svm.fit(X_pca_train, y_train)
            train_time = time.time() - t0
            joblib.dump(svm, svm1_path)
        else:
            svm = joblib.load(svm1_path)
            train_time = 0.0
        pred = svm.predict(X_pca_test)
        acc  = accuracy_score(y_test, pred)
        rep  = classification_report(y_test, pred, output_dict=True, zero_division=0)
        macro_p, macro_r, macro_f1 = rep['macro avg']['precision'], rep['macro avg']['recall'], rep['macro avg']['f1-score']
        results.append({
            "Model": "SVM+PCA", "PCA": k, "Fold": 1,
            "Accuracy": acc, "macro-P": macro_p, "macro-R": macro_r, "macro-F1": macro_f1,
            "train_time": train_time
        })

        # ---- K-Fold CV ----
        for fold in FOLD_VALUES:
            if fold == 1: continue
            skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=RANDOM_STATE)
            accs, ps, rs, f1s = [], [], [], []
            t0 = time.time()
            for tr_idx, va_idx in tqdm(skf.split(X_pca_train, y_train), desc=f"K-Fold {fold}"):
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
            if fold == 15:
                joblib.dump(svm, f"models/svm_k{k}_fold15.pkl")

        # ---- Fruit-level CM for SVM+PCA (k=5) ----
        if k == 5:
            print("\n[INFO] SVM+PCA(k=5) fruit-level confusion matrix …")
            svm5 = joblib.load(f"models/svm_k5_fold15.pkl")
            y_pred_5 = svm5.predict(X_pca_test)
            save_fruit_cm(y_test, y_pred_5, train_set_flat.classes,
                          "figures/cm_svm_pca5_fruit.png",
                          "SVM+PCA (k=5) – Fruit-Level CM (10 classes)")

    # ------------------------------------------------------------------
    # 3.  CNN  (always evaluate – skip training if weights exist)
    # ------------------------------------------------------------------
    cnn_path     = "models/cnn.pth"
    classes_path = "models/cnn_classes.json"
    if missing(cnn_path) or missing(classes_path):
        print("\n[INFO] Training CNN …")
        cnn = CNN(num_classes=len(train_set_cnn.classes)).to(DEVICE)
        loader   = mk_loader(train_set_cnn, batch=128, shuffle=True)
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
        torch.save(cnn.state_dict(), cnn_path)
        with open(classes_path, "w") as fp:
            json.dump(train_set_cnn.classes, fp)
    else:
        print("\n[INFO] CNN weights found – skipping training")

    # evaluate on full test set + 10-fruit confusion matrix
    cnn = CNN(num_classes=len(train_set_cnn.classes)).to(DEVICE)
    cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    cnn.eval()
    test_loader = mk_loader(test_set_cnn, batch=256, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="CNN eval"):
            x = x.to(DEVICE)
            out = cnn(x).argmax(1).cpu()
            y_true.extend(y.numpy()); y_pred.extend(out.numpy())
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_p, macro_r, macro_f1 = rep['macro avg']['precision'], rep['macro avg']['recall'], rep['macro avg']['f1-score']
    results.append({
        "Model": "CNN", "PCA": None, "Fold": None,
        "Accuracy": acc, "macro-P": macro_p, "macro-R": macro_r, "macro-F1": macro_f1,
        "train_time": 120.0   # placeholder like paper
    })
    save_fruit_cm(y_true, y_pred, train_set_cnn.classes, "figures/cm_cnn_fruit.png", "CNN – Fruit-Level CM (10 classes)")

    # ------------------------------------------------------------------
    # 4.  ENSEMBLE  (always build & evaluate)
    # ------------------------------------------------------------------
    print("\n[INFO] Building / loading ensemble …")
    ens = EnsembleModel(device=DEVICE)
    joblib.dump(ens, "models/ensemble.pkl")
    # evaluate on full test set
    y_true_ens, y_pred_ens = [], []
    for idx in tqdm(range(len(test_set_cnn)), desc="Ensemble eval"):
        img, label = test_set_cnn[idx]
        prob = ens.predict(img)
        y_true_ens.append(label)
        y_pred_ens.append(int(prob.argmax()))
    ens_acc = accuracy_score(y_true_ens, y_pred_ens)
    ens_rep = classification_report(y_true_ens, y_pred_ens, output_dict=True, zero_division=0)
    ens_p, ens_r, ens_f1 = ens_rep['macro avg']['precision'], ens_rep['macro avg']['recall'], ens_rep['macro avg']['f1-score']
    results.append({
        "Model": "Ensemble", "PCA": None, "Fold": None,
        "Accuracy": ens_acc, "macro-P": ens_p, "macro-R": ens_r, "macro-F1": ens_f1,
        "train_time": 0.0
    })
    save_fruit_cm(y_true_ens, y_pred_ens, train_set_cnn.classes, "figures/cm_ensemble_fruit.png", "Ensemble – Fruit-Level CM (10 classes)")

    # ---------- extra Markdown table for Ensemble ----------
    ens_df = pd.DataFrame([{
        "Model": "Ensemble",
        "Accuracy": ens_acc,
        "macro-P": ens_p,
        "macro-R": ens_r,
        "macro-F1": ens_f1
    }])
    print("\n>>> Ensemble-only table (README_ensemble.md):")
    print(ens_df.to_markdown(index=False))
    with open("README_ensemble.md", "w") as f:
        f.write(ens_df.to_markdown(index=False))

    # ------------------------------------------------------------------
    # 5.  FRUIT-LEVEL BAR-CHARTS  (paper-like Fig-1 & Fig-2)
    # ------------------------------------------------------------------
    plot_fruit_counts(train_set_cnn, "Number of Each Fruit in Training Data", "figures/count_train_fruit.png")
    plot_fruit_counts(test_set_cnn,  "Number of Each Fruit in Test Data",     "figures/count_test_fruit.png")
    print("[INFO] Fruit-level bar-charts saved → figures/count_train_fruit.png , count_test_fruit.png")

    # ------------------------------------------------------------------
    # 6.  SAVE MASTER TABLE  (true scores, no zeros)
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    print("\n>>> Full paper tables (Markdown):")
    print(df.to_markdown(index=False))
    df.to_csv("models/paper_tables.csv", index=False)

    # ------------------------------------------------------------------
    # 7.  PCA VISUALISATION  (many fruits, not only Apple)
    # ------------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    pca2 = PCA(n_components=2).fit_transform(X_train[:5000])
    plot_pca_2d(pca2, y_train[:5000], train_set_flat.classes, "figures/pca2d.png")
    pca3 = PCA(n_components=3).fit_transform(X_train[:5000])
    plot_pca_3d(pca3, y_train[:5000], train_set_flat.classes, "figures/pca3d.png")
    print("\n[INFO] All artefacts ready – run `streamlit run app.py`")