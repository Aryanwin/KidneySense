import torch
import torchvision
from torchvision import transforms, utils
import torchvision.transforms.functional as fn
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import torchvision.datasets as Datasets
import torch.optim as optim
from   sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from   sklearn.model_selection import train_test_split
from collections import OrderedDict
from  sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import os
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from random import shuffle
from math import floor
from PIL import Image

import cv2
import imghdr

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import glob

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
    self.batchnorm1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
    self.batchnorm2 = nn.BatchNorm2d(8)
    self.pool2 = nn.MaxPool2d(2)
    self.conv3 = nn.Conv2d(8, 32, 3, padding=1)
    self.batchnorm3 = nn.BatchNorm2d(32)
    self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
    self.batchnorm4 = nn.BatchNorm2d(32)
    self.pool4 = nn.MaxPool2d(2)
    self.conv5 = nn.Conv2d(32, 128, 3, padding=1)
    self.batchnorm5 = nn.BatchNorm2d(128)
    self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
    self.batchnorm6 = nn.BatchNorm2d(128)
    self.pool6 = nn.MaxPool2d(2)
    self.conv7 = nn.Conv2d(128, 4, 1) #changed to account for 4 channels instead of 10
    self.pool7 = nn.AvgPool2d(3)

  def forward(self, x):
    #-------------
    # INPUT
    #-------------
    print(x.shape)
    #print(x)
    #x.toTensor
    x = x.reshape(-1, 3, 32, 32)

    #-------------
    # LAYER 1
    #-------------
    output_1 = self.conv1(x)
    output_1 = F.relu(output_1)
    output_1 = self.batchnorm1(output_1)

    #-------------
    # LAYER 2
    #-------------
    output_2 = self.conv2(output_1)
    output_2 = F.relu(output_2)
    output_2 = self.pool2(output_2)
    output_2 = self.batchnorm2(output_2)

    #-------------
    # LAYER 3
    #-------------
    output_3 = self.conv3(output_2)
    output_3 = F.relu(output_3)
    output_3 = self.batchnorm3(output_3)

    #-------------
    # LAYER 4
    #-------------
    output_4 = self.conv4(output_3)
    output_4 = F.relu(output_4)
    output_4 = self.pool4(output_4)
    output_4 = self.batchnorm4(output_4)

    #-------------
    # LAYER 5
    #-------------
    output_5 = self.conv5(output_4)
    output_5 = F.relu(output_5)
    output_5 = self.batchnorm5(output_5)

    #-------------
    # LAYER 6
    #-------------
    output_6 = self.conv6(output_5)
    output_6 = F.relu(output_6)
    output_6 = self.pool6(output_6)
    output_6 = self.batchnorm6(output_6)

    #--------------
    # OUTPUT LAYER
    #--------------
    output_7 = self.conv7(output_6)


    output_7 = self.pool7(output_7)

    output_7 = output_7.view(-1, 4) # changed to output a vector of length 4


    return F.softmax(output_7, dim=1)