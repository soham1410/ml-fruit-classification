"""
src/cnn.py
~~~~~~~~~~
6-layer CNN **exactly** as described in the DAML 2023 paper, plus
light-weight training and evaluation helpers.

Paper architecture (Fig-6):
    Input  3@100×100
    Conv1  8@100×100  + ReLU + MaxPool2d(2)  →  8@50×50
    Conv2 16@50×50    + ReLU + MaxPool2d(2)  → 16@25×25
    Conv3 32@25×25    + ReLU + MaxPool2d(2)  → 32@12×12  (floor division)
    Flatten
    FC     512        + ReLU + Dropout(0.2)
    Output num_classes (131 for Fruit-360)
"""
# -------------------- imports --------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# ------------------------------------------------------------------
# 1. Model definition
# ------------------------------------------------------------------
class CNN(nn.Module):
    """
    6-layer CNN identical to the paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes (131 for Fruit-360).
    """
    def __init__(self, num_classes: int = 131):
        super().__init__()
        # ---- feature extractor ----
        self.features = nn.Sequential(
            # Conv1  3→8  100×100  →  50×50
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # keep spatial size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Conv2  8→16  50×50  →  25×25
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Conv3  16→32  25×25  →  12×12
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # ---- classifier ----
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
# ------------------------------------------------------------------
# 2. Training helper
# ------------------------------------------------------------------
def train_cnn(net: CNN,
              dataset,
              epochs: int = 6,
              batch: int = 128,
              lr: float = 1e-3,
              device: torch.device = torch.device("cpu")):
    """
    Simple training loop that reproduces the paper’s 6-epoch schedule.

    Parameters
    ----------
    net : CNN
        Model instance (already on target device).
    dataset : torch.utils.data.Dataset
        Training split (e.g. Fruit360(train=True, transform=...)).
    epochs : int
    batch : int
    lr : float
    device : torch.device
    """
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(1, epochs + 1):
        bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        running_loss = 0.0
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            bar.set_postfix(loss=loss.item())
        print(f"[INFO] Epoch {epoch} – avg loss = {running_loss/len(loader):.4f}")
# ------------------------------------------------------------------
# 3. Evaluation helper
# ------------------------------------------------------------------
def eval_cnn(net: CNN, dataset, device: torch.device = torch.device("cpu")):
    """
    Evaluate model and return metrics used in the paper.

    Returns
    -------
    accuracy : float
    macro_precision : float
    macro_recall : float
    macro_f1 : float
    """
    loader = DataLoader(dataset, batch_size=256, num_workers=4, pin_memory=True)
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = net(x).argmax(1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(out.numpy())
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_p = rep['macro avg']['precision']
    macro_r = rep['macro avg']['recall']
    macro_f1 = rep['macro avg']['f1-score']
    return acc, macro_p, macro_r, macro_f1