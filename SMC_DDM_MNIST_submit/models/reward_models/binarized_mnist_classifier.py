import torch
import torch.nn as nn
import torch.nn.functional as F

class BinarizedMNISTClassifier(nn.Module):
    def __init__(self):
        super(BinarizedMNISTClassifier, self).__init__()
        self.embedding = nn.Embedding(2, 16)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3)   # Output: (B, 32, 26, 26)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # Output: (B, 64, 24, 24)
        self.pool1 = nn.MaxPool2d(2, 2)                 # Output: (B, 64, 12, 12)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # Output: (B, 128, 10, 10)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3) # Output: (B, 256, 8, 8)
        self.pool2 = nn.MaxPool2d(2, 2)                 # Output: (B, 256, 4, 4)
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        x.shape: (B, 1, 28, 28, 2)
        """
        x = x @ self.embedding.weight # Shape: (B, 1, 28, 28, 16)
        x = x.transpose(1, -1).squeeze(-1) # Shape: (B, 16, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.flatten(start_dim=1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Shape: (B, 10)


class SimpleBinarizedMNISTClassifier(nn.Module):
    def __init__(self):
        super(SimpleBinarizedMNISTClassifier, self).__init__()
        self.fc = nn.Linear(28 * 28 * 2, 10)

    def forward(self, x):
        """
        x.shape: (B, 1, 28, 28, 2)
        """
        x = self.fc(x.flatten(start_dim=1))
        return x  # Shape: (B, 10)