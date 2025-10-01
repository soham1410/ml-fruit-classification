class EnsembleModel:
    def __init__(self, device):
        self.device=device
        self.cnn = CNN(num_classes=131).to(device)
        self.cnn.load_state_dict(torch.load("models/cnn.pth", map_location=device))
        self.cnn.eval()
        self.pca = joblib.load("models/pca_k5.pkl")   # best in paper
        self.svm = joblib.load("models/svm_k5_fold15.pkl")
    def predict(self, img_pil):
        # CNN branch
        x_cnn = transforms.ToTensor()(img_pil.resize((100,100))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn(x_cnn).cpu()
            prob_cnn = torch.softmax(logits,1).numpy()[0]
        # SVM+PCA branch
        x_flat = np.array(img_pil.resize((100,100))).reshape(-1)/255.0
        x_pca  = self.pca.transform([x_flat])
        prob_svm = self.svm.predict_proba(x_pca)[0]
        # soft vote
        prob = (prob_cnn + prob_svm)/2
        return prob