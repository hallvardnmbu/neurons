import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(2, 3, kernel_size=2, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(3 * 3 * 3, 5, bias=True)

        self.conv1.weight.data = torch.Tensor([[[[1.0, 1.0], [2.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]],
                                               [[[2.0, 2.0], [1.0, 1.0]], [[2.0, 1.0], [2.0, 1.0]]],
                                               [[[0.0, 0.0], [0.0, 0.0]],
                                                [[0.0, 0.0], [0.0, 0.0]]]])
        self.fc1.weight.data = torch.Tensor([[2.5 for _ in range(3 * 3 * 3)],
                                             [-1.2 for _ in range(3 * 3 * 3)],
                                             [0.5 for _ in range(3 * 3 * 3)],
                                             [3.5 for _ in range(3 * 3 * 3)],
                                             [5.2 for _ in range(3 * 3 * 3)]])
        self.fc1.bias.data = torch.Tensor([3.0, 4.0, 5.0, 6.0, 7.0])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


network = Network()

input = torch.Tensor([[[[0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 2.0, 0.0],
                        [0.0, 3.0, 4.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]],

                       [[0.0, 0.0, 0.0, 0.0],
                        [0.0, 4.0, 3.0, 0.0],
                        [0.0, 2.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]]]])
input.requires_grad = True

output = network(input)
gradient = torch.ones((1, 5))
output.backward(gradient)

# Backpropagate the gradient
print(network.conv1.weight - 0.1 * network.conv1.weight.grad)
print(network.fc1.weight - 0.1 * network.fc1.weight.grad)
print(network.fc1.bias - 0.1 * network.fc1.bias.grad)
