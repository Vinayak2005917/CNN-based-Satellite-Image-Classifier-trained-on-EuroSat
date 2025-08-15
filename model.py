
import torch
import torch.nn as nn

class CNN64x64(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN64x64, self).__init__()

        # ---- Convolutional layers ----
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # 64x64 -> 64x64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)                                      # 64x64 -> 64x64
        self.pool = nn.MaxPool2d(2, 2)                                                   # 64x64 -> 32x32

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)                                     # 32x32 -> 32x32
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)                                    # 32x32 -> 32x32
        self.pool2 = nn.MaxPool2d(2, 2)                                                  # 32x32 -> 16x16

        # ---- Fully connected layers ----
        self.fc1 = nn.Linear(256*16*16, 512)  # flatten
        self.fc2 = nn.Linear(512, num_classes)

        # Activation
        self.relu = nn.ReLU()
        # Optional dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First block
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Second block
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
