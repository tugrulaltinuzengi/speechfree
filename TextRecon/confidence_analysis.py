import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import os
from models import ElectrolarynxCNN


def analyze_confidence_threshold(dataset_path="dataset.npz", model_weights="best_model.pth",
                                 output_json="confidence_threshold.json"):
    """
    Modelin validation/test seti üzerindeki tahminlerini analiz eder,
    doğru ve yanlış tahminlerin Softmax skorlarına göre optimum güven eşiğini (threshold) belirler.
    """
    if not os.path.exists(dataset_path) or not os.path.exists(model_weights):
        print("HATA: Dataset veya Model ağırlıkları bulunamadı. Önce veri derleyip modeli eğitmelisin!")
        return

    print("1. Veri ve Model yükleniyor...")
    data = np.load(dataset_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    y_true = torch.tensor(data['y'], dtype=torch.long)

    model = ElectrolarynxCNN(num_classes=10)
    model.load_state_dict(torch.load(model_weights, map_location='cpu'))
    model.eval()

    print("2. Tüm veri üzerinde Inference yapılıyor...")
    with torch.no_grad():
        logits = model(X)
        probabilities = F.softmax(logits, dim=1)
        max_probs, preds = torch.max(probabilities, dim=1)

    # Numpy array'e çevir
    max_probs = max_probs.numpy()
    preds = preds.numpy()
    y_true = y_true.numpy()

    # 3. Doğru tahminleri (1) ve Yanlış tahminleri (0) ayır
    # Amacımız: Yanlış/Emin olunmayan tahminleri düşük Softmax skoruyla yakalayıp reddetmek.
    correct_mask = (preds == y_true).astype(int)

    print("3. ROC Analizi ve Eşik (Threshold) hesaplanıyor...")
    fpr, tpr, thresholds = roc_curve(correct_mask, max_probs)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic = TPR - FPR (En ideal denge noktası)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]

    print(f"\n--- ANALİZ SONUÇLARI ---")
    print(f"ROC AUC Skoru: {roc_auc:.4f}")
    print(f"Hesaplanan Optimal Confidence Threshold: {optimal_threshold:.4f}")

    # 4. Yiğit'in kullanması için JSON olarak dışa aktar
    config = {
        "confidence_threshold": float(optimal_threshold),
        "description": "If top-1 softmax probability is below this threshold, reject the prediction as noise/swallowing."
    }

    with open(output_json, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\nEşik değeri '{output_json}' dosyasına kaydedildi. (Yiğit'e iletilecek)")

    # 5. Görselleştirme (Histogram ve ROC Eğrisi)
    print("Grafikler çiziliyor...")
    plt.figure(figsize=(12, 5))

    # Alt Grafik 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(max_probs[correct_mask == 1], bins=30, alpha=0.5, color='green', label='Doğru Tahminler')
    plt.hist(max_probs[correct_mask == 0], bins=30, alpha=0.5, color='red', label='Yanlış Tahminler (Gürültü)')
    plt.axvline(x=optimal_threshold, color='blue', linestyle='--', label=f'Optimal Eşik ({optimal_threshold:.2f})')
    plt.title("Softmax Confidence Dağılımı")
    plt.xlabel("Top-1 Olasılık Skoru")
    plt.ylabel("Frekans")
    plt.legend()

    # Alt Grafik 2: ROC Eğrisi
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=50, zorder=5, label=f'Optimal Nokta')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Kabul)')
    plt.ylabel('True Positive Rate (Doğru Kabul)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test amaçlı, model ağırlıkları henüz olmadığı için hata verebilir.
    # Model eğitildikten sonra doğrudan çalıştırılacak.
    pass