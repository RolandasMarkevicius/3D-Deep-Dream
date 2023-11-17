import numpy as np
import math
import matplotlib.pyplot as plt

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
    #print(np_image.shape)
    np_image = np_image.transpose((1, 2, 0))
    axis.axis('off')
    axis.set_title(label = label)
    axis.imshow(np_image)
  plt.show()

plot_batch_samples(train_loader, batch_size = batch_size)