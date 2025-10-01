"""
src/data.py
~~~~~~~~~~~

Fruit-360 data-set wrapper.

- Automatically downloads the data set from Kaggle the **first** time it is
  instantiated (requires Kaggle API key to be configured).
- Exposes the standard PyTorch `ImageFolder` interface, so it can be used with
  any `torchvision.transforms` pipeline.
"""
# -------------------- imports --------------------
import os
import subprocess
import zipfile
from torchvision.datasets import ImageFolder
# ------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------
DEFAULT_ROOT = "data"  # top-level directory for data storage
KAGGLE_SLUG  = "moltean/fruits"  # Kaggle data-set identifier
# ------------------------------------------------------------------
#  Fruit360: thin wrapper around ImageFolder
# ------------------------------------------------------------------
class Fruit360(ImageFolder):
    """
    Fruit-360 data-set loader.

    Parameters
    ----------
    root : str, optional
        Directory where the data set will be stored.
        Defaults to ``data/``.
    train : bool, optional
        If ``True`` returns the **Training** split, otherwise **Test**.
    download : bool, optional
        If ``True`` **and** the data set is not found locally, it will be
        downloaded from Kaggle via the official API.
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a
        transformed version (e.g., `torchvision.transforms.ToTensor()`).
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    **kwargs
        Additional arguments passed to ``ImageFolder``.
    """
    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        train: bool = True,
        download: bool = False,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        self.root = root
        self.split = "Training" if train else "Test"
        self.folder = os.path.join(root, "fruits-360_dataset", "fruits-360", self.split)

        # auto-download if requested and not present
        if download and not os.path.isdir(self.folder):
            self._download()

        super().__init__(
            self.folder,
            transform=transform,
            target_transform=target_transform,
            **kwargs
        )

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _download(self) -> None:
        """
        Downloads and extracts the Fruit-360 data set from Kaggle.

        Requires the Kaggle API key to be configured:
        ``~/.kaggle/kaggle.json`` (see official Kaggle documentation).
        """
        print(f"[INFO] Fruit-360 not found at {self.folder}")
        print("[INFO] Downloading from Kaggle … (this happens only once)")

        # ensure root directory exists
        os.makedirs(self.root, exist_ok=True)

        # download ZIP via Kaggle CLI
        zip_path = os.path.join(self.root, "fruits.zip")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", self.root],
            check=True
        )

        # extract ZIP
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

        # clean-up
        os.remove(zip_path)
        print(f"[INFO] Fruit-360 extracted to {self.root}")