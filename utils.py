import numpy as np
import pandas as pd

import math
import os

import matplotlib.pyplot as plt
from PIL import Image 

import torch
from torchvision import models, transforms
from torch.utils.data import Dataset


# Custom DataSet Class
class CustomDataSet(Dataset):
    def __init__(self, folder_path, csv_path, transforms = None):
        #get all the necessary parameters for directories, labels and transforms
        self.csv = pd.read_csv(csv_path)
        self.folder_path = folder_path
        self.transforms = transforms

    def __len__(self):
        #get the length of the dataset
        class_length = len(self.csv.iloc[:, 1])
        return class_length

    def __getitem__(self, idx):
        #get the dataset item and label and set transforms
        self.image_path_list = [os.path.join(self.folder_path, image_name) for image_name in self.csv.iloc[:, 0]]
        self.one_hot_label = pd.get_dummies(self.csv.iloc[:, 1])
        self.one_hot_label_list = self.one_hot_label.values.tolist()

        self.image_path = self.image_path_list[idx]
        self.image = Image.open(self.image_path)
        self.label = torch.tensor(self.one_hot_label_list[idx]).float()

        if self.transforms:
            self.image = self.transforms(self.image)

        sample = {'image': self.image, 'label': self.label}     

        return sample
    

# Convolutional arithmetic
def conv_arithmetic(input, kernel, stride, padding, dilation):
  output = ((input + 2*padding - dilation * (kernel - 1) - 1) / stride) + 1
  print(output)


  # Visualise batch
def plot_batch_samples(data_loader, batch_size):

  # Set up the figure grid
  log = math.log(batch_size, 2)
  rows = 2**math.ceil(log/2)
  columns = 2**math.ceil(log/2)

  # Set up figure
  fig = plt.figure(figsize = (3*columns, 3*rows), dpi = 72)

    # Get a single batch
  batch = next(iter(data_loader))

  # Loop through the batch samples and labels, and populate them in the figure
  for idx, (image, label) in enumerate(zip(batch['image'], batch['label'])):
    axis = fig.add_subplot(rows, columns, idx+1)
    np_image = image.numpy()
    np_image = np_image.transpose((1, 2, 0))
    axis.axis('off')
    axis.set_title(label = label)
    axis.imshow(np_image)
  plt.show()