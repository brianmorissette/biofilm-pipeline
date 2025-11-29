import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Tiny CNN for 1x128x128 → scalar ----
class SurfaceAreaCNN(nn.Module):
    def __init__(self, image_size=128, first_layer_channels=8, dropout=0.0, weight_decay=0.0):
        """
        Args:
            image_size (int): The height/width of the input image (assumes square).
            first_layer_channels (int): Number of channels in the first conv layer.
            dropout (float): Dropout probability applied after activations (0.0 = no dropout).
            weight_decay (float): Weight decay coefficient to be used in your optimizer (not used in model).
        """
        super().__init__()
        self.weight_decay = weight_decay  # Store for reference; pass explicitly to optimizer for effect
        self.feat = nn.Sequential(
            nn.Conv2d(1, first_layer_channels, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.MaxPool2d(2),                         # 64x64
            nn.Conv2d(first_layer_channels, first_layer_channels * 2, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.MaxPool2d(2),                         # 32x32
            nn.Conv2d(first_layer_channels * 2, first_layer_channels * 4, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.MaxPool2d(2),                         # 16x16
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(first_layer_channels * 4 * image_size // 8 * image_size // 8, 128), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, 1),                     # regression output
        )

    def forward(self, x):
        return self.head(self.feat(x)).squeeze(-1)  # [B]

class TinyCNN(nn.Module):
    def __init__(self, image_size=128):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 64x64
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 16x16
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * image_size // 8 * image_size // 8, 128), nn.ReLU(),
            nn.Linear(128, 1),                     # regression output
        )

    def forward(self, x):
        return self.head(self.feat(x)).squeeze(-1)  # [B]

class FirstCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, image_size=28):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For biofilm images, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case 1 (surface area).
            image_size: int
                The size of the input image.
        """
        super(CNN, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 1 output features (num_classes)
        self.fc1 = nn.Linear(16 * (image_size // 8) * (image_size // 8), num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        return x




