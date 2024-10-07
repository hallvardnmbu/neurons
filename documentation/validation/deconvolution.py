import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the deconvolution layer
class DeconvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DeconvolutionLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.deconv(x)

# Test the backward pass
def test_backward_pass():
    # Define the input parameters
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)

    # Create the deconvolution layer
    layer = DeconvolutionLayer(in_channels, out_channels, kernel_size, stride, padding)

    # Define the input tensor
    input_tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0],
                                   [9.0, 10.0, 11.0, 12.0],
                                   [13.0, 14.0, 15.0, 16.0]]]], requires_grad=True)

    # Define the gradient tensor
    grad_tensor = torch.tensor([[[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                  [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                                  [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3],
                                  [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
                                  [3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9],
                                  [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7],
                                  [4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5]]]])

    # Forward pass
    output = layer(input_tensor)

    # Backward pass
    output.backward(grad_tensor)

    # Get the gradients
    input_grad = input_tensor.grad
    weight_grad = layer.deconv.weight.grad

    # Check the shapes
    assert input_grad.shape == input_tensor.shape
    assert weight_grad.shape == layer.deconv.weight.shape

    print("Input gradient shape:", input_grad.shape)
    print("Weight gradient shape:", weight_grad.shape)

    print("Kernel gradient:\n", weight_grad)

# Run the test
test_backward_pass()
