import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceAreaCNN(nn.Module):
    """
    CNN model for predicting biofilm surface area from release images.
    Designed for 128x128 input images.
    """

    def __init__(
        self, image_size=128, first_layer_channels=8, dropout=0.0, weight_decay=0.0
    ):
        """
        Args:
            image_size: The height/width of the input image (assumes square).
            first_layer_channels: Number of channels in the first conv layer.
            dropout: Dropout probability applied after activations.
            weight_decay: Weight decay coefficient (stored for reference).
        """
        super().__init__()
        self.weight_decay = weight_decay
        self.feat = nn.Sequential(
            nn.Conv2d(1, first_layer_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(first_layer_channels, first_layer_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(first_layer_channels * 2, first_layer_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.MaxPool2d(2),  # 16x16
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                first_layer_channels * 4 * image_size // 8 * image_size // 8, 128
            ),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, 1),  # regression output
        )

    def forward(self, x):
        """Forward pass."""
        return self.head(self.feat(x)).squeeze(-1)  # [B]


class TinyCNN(nn.Module):
    """
    A smaller CNN architecture.
    """

    def __init__(self, image_size=128):
        """
        Args:
            image_size: Input image size.
        """
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * image_size // 8 * image_size // 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # regression output
        )

    def forward(self, x):
        """Forward pass."""
        return self.head(self.feat(x)).squeeze(-1)  # [B]


class FirstCNN(nn.Module):
    """
    Basic CNN architecture.
    """

    def __init__(self, in_channels=1, num_classes=1, image_size=28):
        """
        Define the layers of the convolutional neural network.

        Args:
            in_channels: The number of channels in the input image.
            num_classes: The number of classes to predict.
            image_size: The size of the input image.
        """
        super().__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 1 output features (num_classes)
        # Note: The 16*7*7 assumes image_size=28. For other sizes this needs adjustment.
        self.fc1 = nn.Linear(16 * (image_size // 8) * (image_size // 8), num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc1(x)
        return x


class SimpleFeedForwardNN(nn.Module):
    """
    Simple Feed Forward Neural Network.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        """
        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass."""
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class myNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x