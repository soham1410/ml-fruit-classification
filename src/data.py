import os, torch
from   torchvision.datasets import ImageFolder
from   torchvision import transforms
URL  = "https://www.kaggle.com/datasets/moltean/fruits"
class Fruit360(ImageFolder):
    """Tiny wrapper that downloads Fruit-360 if not present."""
    def __init__(self, root="data", train=True, **kw):
        split = "Training" if train else "Test"
        super().__init__(os.path.join(root,"fruits-360_dataset","fruits-360",split), **kw)