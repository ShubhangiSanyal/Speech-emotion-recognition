import torch
import torchaudio
import torch.nn as nn
from typing import Dict

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        # Define the convolutional blocks using 1D convolutions
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # Maintain dimension
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Halves the sequence length
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Transformer encoder setup
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=4
        )
        
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)

        # Pass through convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Prepare for transformer encoder
        x = x.permute(2, 0, 1) 

        # Transformer encoder
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=0)  # Average over sequence

        # Final fully connected layer
        x = self.fc(x)
        return x
    
class EmotionRecognizer:
    def __init__(self, model_path: str, emotions: Dict[int, str]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.emotions = emotions

    def _load_model(self, model_path: str):
        model = AudioClassifier(num_classes=7)  # Create an instance of your model architecture
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def recognize_emotion(self, audio_path: str):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            output = self.model(waveform)
            pred_idx = output.argmax().item()
            pred_emotion = self.emotions[pred_idx]

        return pred_emotion