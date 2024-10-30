import torch
import torch.nn as nn
import numpy as np

# Create input with same dimensions as your test
in_channels = 3
out_channels = 2
input_h, input_w = 32, 32
kernel_size = 3
stride = 1
padding = 1

# Create conv layer
conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride=stride, padding=padding, bias=False)

# Set weights to match your test (i as f32 for each kernel)
with torch.no_grad():
    for i in range(out_channels):
        conv.weight[i].fill_(1.0 + float(i))

# Create input tensor matching yours (all 0.1)
x = torch.full((1, in_channels, input_h, input_w), 0.1, requires_grad=True)

# Forward pass
output = conv(x)

# Create gradient (ones)
gradient = torch.ones_like(output)
output.backward(gradient)

print("Kernel gradient shape:", conv.weight.grad.shape)

# Print details for comparison
np.set_printoptions(threshold=np.inf)
print("\nOutput:", output.detach().numpy())
print("\nKernel gradients:", conv.weight.grad)
print("\nInput gradients:", x.grad.detach().numpy())
