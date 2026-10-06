import torch
import torch.nn as nn

class ElectrolarynxCNN(nn.Module):
    """
    Birincil Model: 1D-CNN (5 Kanallı Revize Edilmiş Versiyon)
    Giriş: (Batch, 5, 1000)  <-- 6'dan 5'e düştü
    Çıkış: (Batch, 10)
    """

    def __init__(self, num_classes=10):
        super(ElectrolarynxCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: Giriş kanalı (in_channels) 6 yerine 5 yapıldı
            nn.Conv1d(5, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Block 4
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 5, 1000) <-- 5 kanal bekleniyor
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


class ElectrolarynxLSTM(nn.Module):
    """
    Alternatif Model: BiLSTM (5 Kanallı Revize Edilmiş Versiyon)
    Giriş: (Batch, 5, 1000) <-- 6'dan 5'e düştü
    Çıkış: (Batch, 10)
    """

    def __init__(self, num_classes=10):
        super(ElectrolarynxLSTM, self).__init__()
        # input_size=6 yerine 5 yapıldı
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Giriş: (B, 5, 1000) -> (B, 1000, 5)
        x = x.permute(0, 2, 1)

        output, (hn, cn) = self.lstm(x)

        # Son zaman adımlarını birleştir
        last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        x = self.classifier(last_hidden)
        return x