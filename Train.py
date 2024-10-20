import torch.nn as nn
import numpy as np
from ConvNet import ConvNet
from matplotlib import pyplot as plt
from ConvolutionalNueralNetwork import ConvolutionalNeuralNet
from DataOrganization import transformed_dataset_train, transformed_dataset_val


epoch = 1

model = ConvolutionalNeuralNet(ConvNet())

log_dict = model.train(nn.CrossEntropyLoss(), epochs=epoch, batch_size=64,
                       training_set=transformed_dataset_train, validation_set=transformed_dataset_val)

x = np.arange(0,epoch,1)
y1=log_dict["training_loss_per_epoch"]
y2=log_dict["validation_loss_per_epoch"]
plt.plot(x, y1, label='train"ing loss')
plt.plot(x, y2, label='validation loss')

plt.title('Loss Chart')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()

plt.show()