"""
src/viz.py
~~~~~~~~~~

PCA visualisation utilities that reproduce **Fig-9** (2-D) and **Fig-10** (3-D)
from the DAML 2023 paper.

Functions
---------
plot_pca_2d(X, y, classes, save_path)
plot_pca_3d(X, y, classes, save_path)
"""
# -------------------- imports --------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ------------------------------------------------------------------
#  2-D scatter (Fig-9)
# ------------------------------------------------------------------
def plot_pca_2d(X: np.ndarray,
                y: np.ndarray,
                classes: list[str],
                save_path: str):
    """
    Parameters
    ----------
    X : np.ndarray, shape=(N, 2)
        2-D PCA embeddings.
    y : np.ndarray, shape=(N,)
        Integer labels.
    classes : list[str]
        Class names (for legend).
    save_path : str
        Full file path (e.g. "figures/pca2d.png").
    """
    plt.figure(figsize=(6, 5))
    # colour map for up to 9 fruits (matches paper)
    palette = sns.color_palette("tab10", n_colors=len(classes))
    for i, fruit in enumerate(classes[:9]):  # paper shows 9 fruits
        mask = y == i
        plt.scatter(
            X[mask, 0], X[mask, 1],
            s=8, label=fruit, color=palette[i], alpha=0.7
        )
    plt.legend()
    plt.title("PCA = 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
# ------------------------------------------------------------------
#  3-D scatter (Fig-10)
# ------------------------------------------------------------------
def plot_pca_3d(X: np.ndarray,
                y: np.ndarray,
                classes: list[str],
                save_path: str):
    """
    Parameters
    ----------
    X : np.ndarray, shape=(N, 3)
        3-D PCA embeddings.
    y : np.ndarray, shape=(N,)
        Integer labels.
    classes : list[str]
        Class names (for legend).
    save_path : str
        Full file path (e.g. "figures/pca3d.png").
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    palette = sns.color_palette("tab10", n_colors=len(classes))
    for i, fruit in enumerate(classes[:9]):
        mask = y == i
        ax.scatter(
            X[mask, 0], X[mask, 1], X[mask, 2],
            s=8, label=fruit, color=palette[i], alpha=0.7
        )
    ax.legend()
    ax.set_title("PCA = 3")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()