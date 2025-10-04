import joblib
from sklearn.svm import SVC
def train_svm_pca(X, y, k, fold=1):
    pca = joblib.load(f"models/pca_k{k}.pkl")
    X_pca = pca.transform(X)
    svm = SVC(kernel='rbf', gamma='scale', probability=True)
    svm.fit(X_pca, y)
    return svm
def eval_svm_pca(svm, X, y, k):
    pca = joblib.load(f"models/pca_k{k}.pkl")
    X_pca = pca.transform(X)
    return svm.predict(X_pca), svm.predict_proba(X_pca)