import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[:, class_idx].backward()

        # Compute Grad-CAM
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        weights = torch.mean(gradients, dim=[2, 3])  # Global average pooling over width and height

        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize between 0 and 1

        return cam.numpy()