"""
src/ensemble.py
~~~~~~~~~~~~~~~

Soft-voting ensemble that combines:

1. CNN (fine-grained spatial features)
2. SVM + PCA (k=5) – best single PCA+SVM model from paper

The final probability vector is simply the **average** of both
individual probabilities.  This already beats each single model
on the Fruit-360 test split.
"""
# -------------------- imports --------------------
import os
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
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
        # load CNN
        from src.cnn import CNN
        self.cnn = CNN(num_classes=131).to(device)
        self.cnn.load_state_dict(
            torch.load("models/cnn.pth", map_location=device)
        )
        self.cnn.eval()
        # load PCA + SVM (k=5 is best in paper)
        self.pca = joblib.load("models/pca_k5.pkl")
        self.svm = joblib.load("models/svm_k5_fold15.pkl")
        # class names
        with open("models/cnn_classes.json", "r") as fp:
            self.class_names = json.load(fp)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def predict(self, img_pil: Image.Image) -> np.ndarray:
        """
        Parameters
        ----------
        img_pil : PIL.Image
            RGB image of any size.

        Returns
        -------
        prob : np.ndarray  shape=(131,)
            Soft-vote probability vector.
        """
        # ---- CNN branch ----
        x_cnn = transforms.ToTensor()(img_pil.resize((100, 100))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn(x_cnn)
            prob_cnn = F.softmax(logits, dim=1).cpu().numpy()[0]

        # ---- SVM+PCA branch ----
        x_flat = np.array(img_pil.resize((100, 100))).reshape(-1) / 255.0
        x_pca = self.pca.transform([x_flat])
        prob_svm = self.svm.predict_proba(x_pca)[0]

        # ---- soft vote ----
        prob = (prob_cnn + prob_svm) / 2.0
        return prob

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
        loader_flat = DataLoader(flat_dataset, batch_size=256, num_workers=2, pin_memory=True)
        loader_cnn  = DataLoader(cnn_dataset,  batch_size=256, num_workers=2, pin_memory=True)

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