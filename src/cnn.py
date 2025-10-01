import torch.nn as nn
class CNN(nn.Module):
    """Exactly the 6-layer CNN described in paper."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8,16, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*12*12, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self,x): return self.classifier(self.features(x))