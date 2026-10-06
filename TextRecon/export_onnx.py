import torch
import os
from models import ElectrolarynxCNN


def export_model_to_onnx(model_weights_path="best_model.pth", output_path="electrolarynx_classifier.onnx"):
    """
    Eğitilmiş PyTorch modelini mobil cihazlar için ONNX formatına dönüştürür.
    """
    if not os.path.exists(model_weights_path):
        print(f"HATA: {model_weights_path} bulunamadı. Önce modeli eğitmelisin!")
        return

    # 1. Modeli yükle
    model = ElectrolarynxCNN(num_classes=10)
    model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
    model.eval()

    # 2. Sahte (Dummy) bir giriş tensorü oluştur
    # Modelimizin girişi: (Batch_Size, Channels, Sequence_Length) -> (1, 6, 1000)
    dummy_input = torch.randn(1, 5, 1000, requires_grad=False)

    # 3. ONNX olarak dışa aktar
    print(f"Model ONNX formatına dönüştürülüyor... (Opset: 14)")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Android/ONNXRuntime genelde 14'ü çok iyi destekler
        do_constant_folding=True,  # Modeli hızlandırmak için sabitleri birleştirir
        input_names=['input'],  # Yiğit mobilde bu ismi kullanarak tensor besleyecek
        output_names=['output'],  # Çıktı logit'lerinin ismi
        dynamic_axes={
            'input': {0: 'batch_size'},  # Batch size dinamik olabilir
            'output': {0: 'batch_size'}
        }
    )

    print(f"Başarılı! ONNX modeli '{output_path}' olarak kaydedildi.")
    print("Bu dosyayı doğrudan Yiğit'e (Mobil Geliştirici) gönderebilirsin.")


if __name__ == "__main__":
    # Test amaçlı boş bir ağırlık dosyası olmadığı için hata verebilir,
    # ancak model eğitimi bittiğinde direkt çalıştırılabilir.
    pass