from torch import nn
from utils import conv_arithmetic

# Convolutional arithmetic

# conv_arithmetic(input=325, kernel=7, stride=2, padding=0, dilation=1) # out 160 conv
# conv_arithmetic(input=160, kernel=7, stride=2, padding=0, dilation=1) # out 77 maxpool
# conv_arithmetic(input=77, kernel=7, stride=2, padding=0, dilation=1) # out 36 conv
# conv_arithmetic(input=36, kernel=5, stride=2, padding=0, dilation=1) # out 16 maxpool
# conv_arithmetic(input=16, kernel=5, stride=2, padding=0, dilation=1) # out 6 conv
# conv_arithmetic(input=6, kernel=3, stride=1, padding=0, dilation=1) # out 4 conv
# conv_arithmetic(input=4, kernel=3, stride=1, padding=0, dilation=1) # out 2 conv
# conv_arithmetic(input=2, kernel=2, stride=1, padding=0, dilation=1) # out 1 conv
# conv_arithmetic(input=1, kernel=1, stride=1, padding=0, dilation=1) # out 1 conv
# conv_arithmetic(input=1, kernel=1, stride=1, padding=0, dilation=1) # out 1 conv

# Model architecture
class LinearWood(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=7, stride=2, padding=0, dilation=1) #out 160
    self.batchnorm1 = nn.BatchNorm2d(9)
    self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=2, padding=0, dilation=1) # out 77

    self.conv2 = nn.Conv2d(in_channels=9, out_channels=36, kernel_size=7, stride=2, padding=0, dilation=1) # out 36
    self.batchnorm2 = nn.BatchNorm2d(36)
    self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1) # out 16

    self.conv3 = nn.Conv2d(in_channels=36, out_channels=64, kernel_size=5, stride=2, padding=0, dilation=1) # out 6
    self.batchnorm3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1) # out 4
    self.batchnorm4 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, dilation=1) # out 2
    self.batchnorm5 = nn.BatchNorm2d(256)
    self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=0, dilation=1) # out 1
    self.batchnorm6 = nn.BatchNorm2d(512)
    self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, dilation=1) # out 1
    self.batchnorm7 = nn.BatchNorm2d(1024)
    # self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1) # out 1
    #self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1) # out 1

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout2d(0.2)
    self.adaptavgpool = nn.AdaptiveAvgPool2d(output_size=1)

    self.linear_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),      
        nn.Linear(256, 4),
        nn.Softmax(dim=1)
        )

# Forward pass
  def forward(self, x):
    prediction = self.conv1(x)
    prediction = self.batchnorm1(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.maxpool1(prediction)

    prediction = self.conv2(prediction)
    prediction = self.batchnorm2(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.maxpool2(prediction)

    prediction = self.conv3(prediction)
    prediction = self.batchnorm3(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.conv4(prediction)
    prediction = self.batchnorm4(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.conv5(prediction)
    prediction = self.batchnorm5(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.conv6(prediction)
    prediction = self.batchnorm6(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    prediction = self.conv7(prediction)
    prediction = self.batchnorm7(prediction)
    prediction = self.relu(prediction)
    prediction = self.dropout(prediction)

    # prediction = self.conv8(prediction)
    # prediction = self.relu(prediction)
    # prediction = self.dropout(prediction)

    prediction = self.adaptavgpool(prediction)

    prediction = self.linear_stack(prediction)

    return prediction