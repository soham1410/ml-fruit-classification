"""
visualize.py
~~~~~~~~~~~~ 

This script generates all figures for the project from the saved data in
`figures_data.npz`. It can be run independently of `train.py`.
"""
# -------------------- standard library --------------------
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# -------------------- project-specific --------------------
from src.data import Fruit360

# -------------------- helpers --------------------
def clean_fruit_name(cls):
    """Pear_9 → Pear,  Onion_2 → Onion,  Melon_Piel_de_Sapo_1 → Melon"""
    base = cls.split(' ')[0].split('_')[0]
    # ---- optional: drop vegetables / nuts ----
    veg_nuts = {
        'Onion', 'Walnut', 'Quince', 'Cherimoya', 'Beans', 'Beetroot',
        'Cabbage', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber',
        'Eggplant', 'Ginger', 'Kohlrabi', 'Pepper'
    }
    return base if base not in veg_nuts else None

def to_clean_fruits(classes):
    """list of clean names (None → filtered out)"""
    return [clean_fruit_name(c) for c in classes]

def filter_top10_fruits(X, y, classes, top10_names):
    """Filter data to include only the top 10 fruits."""
    clean_labels = [clean_fruit_name(classes[label]) for label in y]
    mask = [label in top10_names for label in clean_labels]
    X_filtered = X[mask]
    y_filtered_str = [clean_labels[i] for i, included in enumerate(mask) if included]
    
    # Create a mapping from fruit name to an integer index
    top10_map = {name: i for i, name in enumerate(top10_names)}
    y_filtered_int = [top10_map[name] for name in y_filtered_str]
    
    return X_filtered, np.array(y_filtered_int), top10_names

# -------------------- plotting functions --------------------
def plot_fruit_counts(targets, classes, title, save_path, top_n=10):
    """Count **real fruits** (no veg/nuts) and plot top-N."""
    cleaned_names = [clean_fruit_name(classes[idx]) for idx in targets]
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
    print(f"[INFO] Saved {save_path}")

def save_fruit_cm(y_true_sub, y_pred_sub, classes, GLOBAL_TOP10_NAMES, GLOBAL_TOP10_IDX, save_path, title="Fruit-Level CM (10 fruits)"):
    """Full 60-class eval → plot only **real fruits** (clean names)."""
    cleaned_fruits = set(to_clean_fruits(classes))
    base_fruits = sorted([f for f in cleaned_fruits if f is not None])
    fruit_to_idx = {f: i for i, f in enumerate(base_fruits)}

    cls_to_fruit_idx = {
        i: fruit_to_idx.get(clean_fruit_name(c), -1) for i, c in enumerate(classes)
    }

    y_true_mapped = np.array([cls_to_fruit_idx[lbl] for lbl in y_true_sub])
    y_pred_mapped = np.array([cls_to_fruit_idx[lbl] for lbl in y_pred_sub])

    mask = y_true_mapped != -1
    y_true_fruit = y_true_mapped[mask]
    y_pred_fruit = y_pred_mapped[mask]

    cm_full = confusion_matrix(y_true_fruit, y_pred_fruit, labels=range(len(base_fruits)))
    
    # Ensure GLOBAL_TOP10_IDX is within bounds
    valid_indices = [idx for idx in GLOBAL_TOP10_IDX if idx < len(base_fruits)]
    cm10 = cm_full[np.ix_(valid_indices, valid_indices)]
    
    # Adjust names to match valid indices
    valid_names = [GLOBAL_TOP10_NAMES[i] for i, idx in enumerate(GLOBAL_TOP10_IDX) if idx < len(base_fruits)]


    plt.figure(figsize=(8, 7))
    sns.heatmap(cm10, annot=True, fmt="d", cmap="Blues",
                xticklabels=valid_names, yticklabels=valid_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Fruit", fontsize=14)
    plt.ylabel("True Fruit", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {save_path}")

def plot_pca_2d(X, y, classes, save_path):
    plt.figure(figsize=(6, 5))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(y)))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=[classes[i] for i in y], s=8, palette=palette, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("PCA = 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved {save_path}")

def plot_pca_3d(X, y, classes, save_path):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    palette = sns.color_palette("tab10", n_colors=len(np.unique(y)))
    
    for i, fruit in enumerate(np.unique(y)):
        mask = y == fruit
        ax.scatter(
            X[mask, 0], X[mask, 1], X[mask, 2],
            s=8, label=classes[fruit], color=palette[i], alpha=0.7
        )
    ax.legend()
    ax.set_title("PCA = 3")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved {save_path}")

# ===================================================================
#  MAIN
# ===================================================================
if __name__ == '__main__':
    os.makedirs("figures", exist_ok=True)

    # --- Load Data ---
    print("[INFO] Loading data from figures_data.npz...")
    try:
        data = np.load("figures_data.npz", allow_pickle=True)
    except FileNotFoundError:
        print("[ERROR] `figures_data.npz` not found. Please run `train.py` first.")
        exit()

    y_test_svm = data["y_test_svm"]
    y_pred_svm = data["y_pred_svm"]
    y_true_cnn = data["y_true_cnn"]
    y_pred_cnn = data["y_pred_cnn"]
    y_true_ens = data["y_true_ens"]
    y_pred_ens = data["y_pred_ens"]
    train_classes = data["train_classes"]
    GLOBAL_TOP10_NAMES = data["GLOBAL_TOP10_NAMES"]
    GLOBAL_TOP10_IDX = data["GLOBAL_TOP10_IDX"]
    X_train_pca = data["X_train_pca"]
    y_train_pca = data["y_train_pca"]
    train_targets = data["train_targets"]
    test_targets = data["test_targets"]

    # --- Generate Plots ---
    print("\n[INFO] Generating plots...")

    # 1. Confusion Matrices
    save_fruit_cm(y_test_svm, y_pred_svm, train_classes, GLOBAL_TOP10_NAMES, GLOBAL_TOP10_IDX,
                  "figures/cm_svm_pca5_fruit.png", "SVM+PCA (k=5) – Fruit-Level CM (10 fruits)")
    save_fruit_cm(y_true_cnn, y_pred_cnn, train_classes, GLOBAL_TOP10_NAMES, GLOBAL_TOP10_IDX,
                  "figures/cm_cnn_fruit.png", "CNN – Fruit-Level CM (10 fruits)")
    save_fruit_cm(y_true_ens, y_pred_ens, train_classes, GLOBAL_TOP10_NAMES, GLOBAL_TOP10_IDX,
                  "figures/cm_ensemble_fruit.png", "Ensemble – Fruit-Level CM (10 fruits)")

    # 2. Fruit Counts
    plot_fruit_counts(train_targets, train_classes, "Number of Each Fruit in Training Data", "figures/count_train_fruit.png")
    plot_fruit_counts(test_targets, train_classes, "Number of Each Fruit in Test Data", "figures/count_test_fruit.png")

    # 3. PCA Visualizations
    X_pca_top10, y_pca_top10, top10_classes = filter_top10_fruits(X_train_pca, y_train_pca, train_classes, GLOBAL_TOP10_NAMES)
    pca2 = PCA(n_components=2).fit_transform(X_pca_top10)
    plot_pca_2d(pca2, y_pca_top10, top10_classes, "figures/pca2d.png")
    pca3 = PCA(n_components=3).fit_transform(X_pca_top10)
    plot_pca_3d(pca3, y_pca_top10, top10_classes, "figures/pca3d.png")

    print("\n[INFO] All figures have been regenerated.")
