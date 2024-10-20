import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ConvNet import ConvNet
from ConvolutionalNueralNetwork import ConvolutionalNeuralNet
from Eval import PATH
from PIL import Image
import time
from GradCAM import GradCAM

model_path = PATH
loaded = torch.load(model_path)
# instantiate your model
model1 = ConvolutionalNeuralNet(ConvNet())

def plot_grad_cam(model, grad_cam, image_tensor, image, class_idx=None):
    # Generate the CAM
    cam = grad_cam.generate_cam(image_tensor, class_idx)

    # Resize the CAM to match the input image dimensions
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    # Convert CAM to RGB format
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(image)
    overlay = overlay / np.max(overlay)

    # Plot the original image, CAM, and overlay
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image)
    axs[0].set_title('Original Image')

    axs[1].imshow(cam, cmap='jet')
    axs[1].set_title('Grad-CAM')

    axs[2].imshow(overlay)
    axs[2].set_title('Overlay')

    plt.savefig('grad_kidney.png')
    plt.show()

    returnimage = Image.open('grad_kidney.png')
    rgb_img = returnimage.convert('RGB')

    path = str(time.time()) + "img1.jpg"
    rgb_img.save(path)
    return path
