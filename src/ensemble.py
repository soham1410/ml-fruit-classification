"""
src/ensemble.py
~~~~~~~~~~~~~~~

Soft-voting ensemble that combines:

1. CNN (fine-grained spatial features)
2. SVM + PCA (k=5) – best single PCA+SVM model from paper

The final probability vector is simply the **average** of both
individual probabilities.  This already beats each single model
on the Fruit-360 test split.

**Fix**: dynamically reads the correct number of classes from
`models/cnn_classes.json` so the checkpoint always matches.
"""
# -------------------- imports --------------------
import os
import json
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.cnn import CNN  # local import

# ------------------------------------------------------------------
#  Device
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
#  EnsembleModel
# ------------------------------------------------------------------
class EnsembleModel:
    """
    Soft-voting ensemble of CNN + SVM+PCA (k=5).

    Usage
    -----
    # training time – fit on full training data (cheap)
    ensemble = EnsembleModel(device=DEVICE)
    ensemble.fit(train_set_flat, train_set_cnn)

    # inference – single image
    prob = ensemble.predict(img_pil)
    label = CLASS_NAMES[prob.argmax()]
    """
    def __init__(self, device: torch.device = DEVICE):
        self.device = device

        # ---------- load class list ----------
        with open("models/cnn_classes.json", "r") as fp:
            self.class_names = json.load(fp)
        num_classes = len(self.class_names)

        # ---------- load CNN (dynamic size) ----------
        self.cnn = CNN(num_classes=num_classes).to(device)
        self.cnn.load_state_dict(
            torch.load("models/cnn.pth", map_location=device)
        )
        self.cnn.eval()

        # ---------- load PCA + SVM (k=5 is best in paper) ----------
        self.pca = joblib.load("models/pca_k5.pkl")
        self.svm = joblib.load("models/svm_k5_fold15.pkl")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def predict(self, img_pil: Image.Image) -> np.ndarray:
        """
        Returns probability vector **aligned to full CNN classes**.
        """
        # ---------- CNN branch ----------
        # inside src/ensemble.py  (predict method)
        img_resized = img_pil.resize((100, 100))
        x_cnn_tensor = transforms.ToTensor()(img_resized)
        x_cnn = x_cnn_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob_cnn = F.softmax(self.cnn(x_cnn), dim=1).cpu().numpy()[0]

        # ---------- SVM+PCA branch (subset classes) ----------
        x_flat = x_cnn_tensor.view(-1).numpy()
        x_pca  = self.pca.transform([x_flat])
        prob_svm = self.svm.predict_proba(x_pca)[0]   # shape [29]

        # ---------- map SVM probs to CNN indices ----------
        # build index map:  svm_class -> cnn_class
        svm_classes = self.svm.classes_          # ndarray(int)  length 29
        cnn_indices = np.arange(len(self.class_names))
        aligned = np.zeros(len(self.class_names), dtype=float)
        for p, svm_idx in zip(prob_svm, svm_classes):
            aligned[svm_idx] = p

        # ---------- soft vote ----------
        return (prob_cnn + aligned) / 2.0

    def score(self, flat_dataset, cnn_dataset):
        """
        Evaluate ensemble accuracy on two parallel data-loaders
        (flattened images for SVM & tensors for CNN).

        Parameters
        ----------
        flat_dataset : torch.utils.data.Dataset
            Returns (flattened_vector, label)
        cnn_dataset  : torch.utils.data.Dataset
            Returns (tensor, label)

        Returns
        -------
        accuracy : float
        """
        from torch.utils.data import DataLoader
        loader_flat = DataLoader(flat_dataset, batch_size=256, num_workers=0, pin_memory=False)
        loader_cnn  = DataLoader(cnn_dataset,  batch_size=256, num_workers=0, pin_memory=False)

        correct = 0
        total   = 0

        for (x_flat, y), (x_cnn, _) in zip(loader_flat, loader_cnn):
            # iterate inside batch
            for i in range(len(y)):
                img = transforms.ToPILImage()(x_cnn[i])
                prob = self.predict(img)
                if prob.argmax() == y[i]:
                    correct += 1
                total += 1

        return correct / total