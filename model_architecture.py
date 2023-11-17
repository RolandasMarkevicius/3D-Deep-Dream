from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import conv_arithmetic

# Convolutional arithmetic
conv_arithmetic(input=32, kernel=5, stride=2, padding=0, dilation=0) # out 16
conv_arithmetic(input=16, kernel=5, stride=2, padding=0, dilation=0) # out 8
conv_arithmetic(input=8, kernel=5, stride=2, padding=0, dilation=0) # out 4

# Model architecture
class LinearWood(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2, padding=0, dilation=1) #126
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1)#62
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2, padding=0, dilation=1) #30
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1)#14
    self.conv3 = nn.Conv2d(in_channels=12, out_channels=36, kernel_size=3, stride=2, padding=0, dilation=1) #6

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout2d(0.2)

    self.linear_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1296, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Dropout(p=0.2),      
        nn.Linear(64, 4),
        nn.Softmax(dim=1)
        )

# Forward pass
  def forward(self, x):
    prediction = self.conv1(x)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.maxpool1(prediction)

    prediction = self.conv2(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.maxpool2(prediction)

    prediction = self.conv3(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.linear_stack(prediction)

    return prediction