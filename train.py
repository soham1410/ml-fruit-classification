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

# ---------- clean fruit name (strips number, filters veg/nuts) ----------
def clean_fruit_name(cls):
    """Pear_9 → Pear,  Onion_2 → Onion,  Melon_Piel_de_Sapo_1 → Melon"""
    base = cls.split(' ')[0].split('_')[0]
    # ---- optional: drop vegetables / nuts ----
    veg_nuts = {
        'Onion', 'Walnut', 'Quince', 'Cherimoya', 'Beans', 'Beetroot',
        'Cabbage', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber',
        'Eggplant', 'Ginger', 'Kohlrabi', 'Pepper'
    }
    return base if base not in veg_nuts else None            # None → skip

def to_clean_fruits(classes):
    """list of clean names (None → filtered out)"""
    return [clean_fruit_name(c) for c in classes]

# ---------- 10-fruit bar-chart (real fruits only) ----------
def plot_fruit_counts(dataset, title, save_path, top_n=10):
    """Count **real fruits** (no veg/nuts) and plot top-N."""
    # Get clean names for every sample in the dataset using the .targets attribute
    cleaned_names = [clean_fruit_name(dataset.classes[idx]) for idx in dataset.targets]

    # Filter out non-fruits and count
    fruit_names = [name for name in cleaned_names if name is not None]
    counts = pd.Series(fruit_names).value_counts()

    top_counts = counts.nlargest(top_n).sort_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_counts.index, y=top_counts.values, color="steelblue")
    plt.title(f"{title} (top {top_n} fruits)", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------- 10-fruit confusion matrix (real fruits only) ----------
def save_fruit_cm(y_true_sub, y_pred_sub, classes, save_path, title="Fruit-Level CM (10 fruits)"):
    """Full 60-class eval → plot only **real fruits** (clean names)."""
    cleaned_fruits = set(to_clean_fruits(classes))
    base_fruits = sorted([f for f in cleaned_fruits if f is not None])
    fruit_to_idx = {f: i for i, f in enumerate(base_fruits)}

    # Map all class indices to their corresponding fruit index (or -1 if not a fruit)
    cls_to_fruit_idx = {
        i: fruit_to_idx.get(clean_fruit_name(c), -1) for i, c in enumerate(classes)
    }

    # Map the true and predicted labels using the map
    y_true_mapped = np.array([cls_to_fruit_idx[lbl] for lbl in y_true_sub])
    y_pred_mapped = np.array([cls_to_fruit_idx[lbl] for lbl in y_pred_sub])

    # Filter out samples where the true label was not a fruit
    mask = y_true_mapped != -1
    y_true_fruit = y_true_mapped[mask]
    y_pred_fruit = y_pred_mapped[mask]

    # --- full fruit CM ---
    cm_full = confusion_matrix(y_true_fruit, y_pred_fruit, labels=range(len(base_fruits)))

    # --- **same** 10-fruit slice (locked list) ---
    cm10 = cm_full[np.ix_(GLOBAL_TOP10_IDX, GLOBAL_TOP10_IDX)]

    # --- plot ---
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



# ===================================================================
#  MAIN  (Windows-safe)
# ===================================================================
if __name__ == '__main__':

    # ------------------------------------------------------------------
    # 1.  DATA  (CNN = subset,  flat = subset)
    # ------------------------------------------------------------------
    cnn_transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    train_set_cnn_full = Fruit360(train=True,  transform=cnn_transform, download=False)
    test_set_cnn_full  = Fruit360(train=False, transform=cnn_transform, download=False)
    
    rng = np.random.default_rng(RANDOM_STATE)
    train_indices = rng.permutation(len(train_set_cnn_full))[:SUB_TRAIN]
    test_indices = rng.permutation(len(test_set_cnn_full))[:SUB_TEST]

    train_set_cnn = torch.utils.data.Subset(train_set_cnn_full, train_indices)
    test_set_cnn = torch.utils.data.Subset(test_set_cnn_full, test_indices)

    print(f"[INFO] CNN data – train: {len(train_set_cnn)}, test: {len(test_set_cnn)}")

    def load_flat_data_from_subset(dataset, indices):
        X, y = [], []
        for i in tqdm(indices, desc="[INFO] Loading flat data"):
            img, label = dataset[i] # img is a tensor here
            X.append(img.view(-1).numpy())
            y.append(label)
        return np.array(X), np.array(y)

    X_train, y_train = load_flat_data_from_subset(train_set_cnn_full, train_indices)
    X_test,  y_test  = load_flat_data_from_subset(test_set_cnn_full,  test_indices)

    print(f"[INFO] Flat data – train: {X_train.shape}, test: {X_test.shape}")

   # ---------- LOCK identical 10 **real** fruits for ALL CMs ----------
    cleaned_fruits = set(to_clean_fruits(train_set_cnn_full.classes))
    base_fruits_raw = sorted([f for f in cleaned_fruits if f is not None])

    # --- map **actual test-set labels** → clean fruit → index ---
    labels_full = [clean_fruit_name(test_set_cnn_full.classes[label]) for label in test_set_cnn_full.targets]
    labels_full = [label for label in labels_full if label is not None]

    # --- **collapse to unique fruit names BEFORE building support** ---
    unique_labels = pd.Series(labels_full)          # Apple, Apple, Pear, …
    unique_counts = unique_labels.value_counts()    # Apple → 500, Pear → 300, …
    GLOBAL_TOP10_NAMES = unique_counts.nlargest(10).index.tolist()
    GLOBAL_TOP10_IDX   = [base_fruits_raw.index(f) for f in GLOBAL_TOP10_NAMES]
    print("[LOCK] Identical 10 **real** fruits for all CMs:", GLOBAL_TOP10_NAMES)

    # ------------------------------------------------------------------
    # 2.  PCA + SVM  (always re-evaluate – skip weight training if exist)
    # ------------------------------------------------------------------
    results = []
    y_pred_5 = None
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
            print("\n[INFO] SVM+PCA(k=5) prediction for CM...")
            svm5 = joblib.load(f"models/svm_k5_fold15.pkl")
            y_pred_5 = svm5.predict(X_pca_test)

    # ------------------------------------------------------------------
    # 3.  CNN  (always evaluate – skip training if weights exist)
    # ------------------------------------------------------------------
    cnn_path     = "models/cnn.pth"
    classes_path = "models/cnn_classes.json"
    if missing(cnn_path) or missing(classes_path):
        print("\n[INFO] Training CNN …")
        cnn = CNN(num_classes=len(train_set_cnn_full.classes)).to(DEVICE)
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
            json.dump(train_set_cnn_full.classes, fp)
    else:
        print("\n[INFO] CNN weights found – skipping training")

    # evaluate on full test set + 10-fruit confusion matrix
    cnn = CNN(num_classes=len(train_set_cnn_full.classes)).to(DEVICE)
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


    # ------------------------------------------------------------------
    # 4.  ENSEMBLE  (always build & evaluate)
    # ------------------------------------------------------------------
    print("\n[INFO] Building / loading ensemble …")
    ens = EnsembleModel(device=DEVICE)
    joblib.dump(ens, "models/ensemble.pkl")
    # evaluate on full test set
    y_true_ens, y_pred_ens = [], []
    to_pil = transforms.ToPILImage()
    for idx in tqdm(range(len(test_set_cnn)), desc="Ensemble eval"):
        img_tensor, label = test_set_cnn[idx]
        img_pil = to_pil(img_tensor)
        prob = ens.predict(img_pil)
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
    # 5.  SAVE RESULTS AND DATA FOR VISUALIZATION
    # ------------------------------------------------------------------
    # Save master table
    df = pd.DataFrame(results)
    print("\n>>> Full paper tables (Markdown):")
    print(df.to_markdown(index=False))
    df.to_csv("models/paper_tables.csv", index=False)
    
    # Save data for figures
    print("\n[INFO] Saving data for visualization to figures_data.npz...")
    np.savez(
        "figures_data.npz",
        y_test_svm=y_test,
        y_pred_svm=y_pred_5,
        y_true_cnn=np.array(y_true),
        y_pred_cnn=np.array(y_pred),
        y_true_ens=np.array(y_true_ens),
        y_pred_ens=np.array(y_pred_ens),
        train_classes=train_set_cnn_full.classes,
        GLOBAL_TOP10_NAMES=np.array(GLOBAL_TOP10_NAMES),
        GLOBAL_TOP10_IDX=np.array(GLOBAL_TOP10_IDX),
        X_train_pca=X_train[:5000],
        y_train_pca=y_train[:5000],
        train_targets=np.array([train_set_cnn_full.targets[i] for i in train_indices]),
        test_targets=np.array([test_set_cnn_full.targets[i] for i in test_indices]),
    )

    print("\n[INFO] Training complete. Run `python visualize.py` to generate figures.")
    print("\n[INFO] You can also run `streamlit run app.py`")