from DataOrganization import transformed_dataset_test
from GradCamPlot import plot_grad_cam, model1
from GradCAM import GradCAM
import torch
#from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#instantiate gradCAM object
grad_cam = GradCAM(model1.network, model1.network.conv6)

def gradout(image):
    image = image.to(torch.float)
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return plot_grad_cam(model1.network, grad_cam, image, image_np)

