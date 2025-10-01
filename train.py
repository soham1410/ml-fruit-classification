"""
train.py
~~~~~~~~
Trains every model described in the paper and dumps:
  - PCA object              → models/pca_k{k}.pkl
  - SVM object              → models/svm_k{k}_fold{f}.pkl
  - CNN state-dict           → models/cnn.pth
  - CNN class-names          → models/cnn_classes.json
  - Ensemble weights         → models/ensemble_weights.json
After running once you can freely launch `streamlit run app.py`.
"""
import os, json, joblib, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.decomposition import PCA
from  sklearn.svm          import SVC
from  sklearn.model_selection import StratifiedKFold
from  sklearn.metrics import accuracy_score, classification_report
import torch, torchvision
from   torch.utils.data import DataLoader
from   torchvision import transforms
from   src.data import Fruit360
from   src.pca_svm import train_svm_pca, eval_svm_pca
from   src.cnn     import CNN, train_cnn, eval_cnn
from   src.ensemble import EnsembleModel
from   src.viz import plot_pca_2d, plot_pca_3d
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------
# 1.  Hyper-parameters  (exactly like paper)
# -------------------------------------------------
PCA_COMPONENTS = [2, 5, 8]   # Table 1,2,3,4
K_FOLD         = 15          # best in paper
RANDOM_STATE   = 42
# -------------------------------------------------
# 2.  Data
# -------------------------------------------------
train_set = Fruit360(train=True,  download=True, transform=transforms.ToTensor())
test_set  = Fruit360(train=False, download=True, transform=transforms.ToTensor())
# For PCA+SVM we need flat vectors – resize to 100×100 like paper
flat_transform = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1).numpy())
])
train_set_flat = Fruit360(train=True,  download=False, transform=flat_transform)
test_set_flat  = Fruit360(train=False, download=False, transform=flat_transform)
# -------------------------------------------------
# 3.  PCA + SVM  (Tables 1,2,3,4)
# -------------------------------------------------
print("=== PCA+SVM training (may take a few minutes) ===")
X_train = np.array([x for x, _ in train_set_flat])
y_train = np.array([y for _, y in train_set_flat])
X_test  = np.array([x for x, _ in test_set_flat])
y_test  = np.array([y for _, y in test_set_flat])

results = []   # collect rows for markdown table
for k in PCA_COMPONENTS:
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test  = pca.transform(X_test)
    joblib.dump(pca, f"models/pca_k{k}.pkl")
    # Table 1:  SVM+PCA  (no CV)
    svm = SVC(kernel='rbf', gamma='scale', random_state=RANDOM_STATE)
    t0 = time.time()
    svm.fit(X_pca_train, y_train)
    train_time = time.time() - t0
    pred = svm.predict(X_pca_test)
    acc  = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    macro_p  = report['macro avg']['precision']
    macro_r  = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    results.append(["SVM+PCA", k, 1, acc, macro_p, macro_r, macro_f1, train_time])
    joblib.dump(svm, f"models/svm_k{k}_fold1.pkl")
    # Tables 2,3,4:  SVM+PCA+K-Fold
    for fold in [2, 5, 10, 15]:
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=RANDOM_STATE)
        fold_accs, fold_ps, fold_rs, fold_f1s = [], [], [], []
        t0 = time.time()
        for tr_idx, va_idx in skf.split(X_pca_train, y_train):
            svm_fold = SVC(kernel='rbf', gamma='scale', random_state=RANDOM_STATE)
            svm_fold.fit(X_pca_train[tr_idx], y_train[tr_idx])
            pred_fold = svm_fold.predict(X_pca_train[va_idx])
            rep = classification_report(y_train[va_idx], pred_fold, output_dict=True, zero_division=0)
            fold_accs.append(accuracy_score(y_train[va_idx], pred_fold))
            fold_ps.append(rep['macro avg']['precision'])
            fold_rs.append(rep['macro avg']['recall'])
            fold_f1s.append(rep['macro avg']['f1-score'])
        train_time = time.time() - t0
        acc  = np.mean(fold_accs)
        macro_p  = np.mean(fold_ps)
        macro_r  = np.mean(fold_rs)
        macro_f1 = np.mean(fold_f1s)
        results.append(["SVM+PCA+KF", k, fold, acc, macro_p, macro_r, macro_f1, train_time])
        # save best fold=15 model
        if fold == 15:
            best_svm = SVC(kernel='rbf', gamma='scale', random_state=RANDOM_STATE)
            best_svm.fit(X_pca_train, y_train)
            joblib.dump(best_svm, f"models/svm_k{k}_fold15.pkl")

df = pd.DataFrame(results, columns=["Model","PCA","Fold","Accuracy","macro-P","macro-R","macro-F1","train_time"])
print(df.to_markdown(index=False))
df.to_csv("models/paper_tables.csv", index=False)

# -------------------------------------------------
# 4.  CNN  (Table 5)
# -------------------------------------------------
print("=== CNN training (≈2 min on GPU) ===")
cnn_transform = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor()
])
train_set_cnn = Fruit360(train=True,  download=False, transform=cnn_transform)
test_set_cnn  = Fruit360(train=False, download=False, transform=cnn_transform)
cnn_model = CNN(num_classes=len(train_set_cnn.classes)).to(device)
train_cnn(cnn_model, train_set_cnn, epochs=6, batch=128, lr=0.001, device=device)
torch.save(cnn_model.state_dict(), "models/cnn.pth")
with open("models/cnn_classes.json","w") as fp:
    json.dump(train_set_cnn.classes, fp)
acc, macro_p, macro_r, macro_f1 = eval_cnn(cnn_model, test_set_cnn, device=device)
print(f"CNN -> Accuracy:{acc:.3f}  macro-P:{macro_p:.3f}  macro-R:{macro_r:.3f}  macro-F1:{macro_f1:.3f}")

# -------------------------------------------------
# 5.  Ensemble  (improvement)
# -------------------------------------------------
print("=== Ensemble training ===")
ensemble = EnsembleModel(device=device)
ensemble.fit(train_set_flat, train_set_cnn)   # needs both flat & CNN datasets
joblib.dump(ensemble, "models/ensemble.pkl")
acc = ensemble.score(test_set_flat, test_set_cnn)
print(f"Ensemble Accuracy = {acc:.3f}")

# -------------------------------------------------
# 6.  PCA visualisation (Fig 9 & 10)
# -------------------------------------------------
pca_2d = PCA(n_components=2).fit_transform(X_train[:5000])   # subsample for speed
plot_pca_2d(pca_2d, y_train[:5000], train_set_flat.classes, save="figures/pca2d.png")
pca_3d = PCA(n_components=3).fit_transform(X_train[:5000])
plot_pca_3d(pca_3d, y_train[:5000], train_set_flat.classes, save="figures/pca3d.png")
print("PCA plots saved under figures/")
print("All done – launch `streamlit run app.py`")