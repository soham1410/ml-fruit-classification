def plot_pca_2d(X, y, classes, save):
    plt.figure(figsize=(6,5))
    for i, fruit in enumerate(classes[:9]):   # paper shows 9 fruits
        mask = y==i
        plt.scatter(X[mask,0], X[mask,1], s=8, label=fruit)
    plt.legend()
    plt.title("PCA=2")
    plt.tight_layout()
    plt.savefig(save); plt.close()