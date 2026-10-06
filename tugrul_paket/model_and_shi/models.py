import torch
import torch.nn as nn

class ElectrolarynxCNN(nn.Module):
    """1D-CNN, 3 kanal PoC versiyonu. Giriş (B, 3, 1000) -> (B, 10)"""
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


class ElectrolarynxLSTM(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.classifier(last_hidden)
