import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(2, 3, kernel_size=2, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(3 * 3 * 3, 5, bias=True)

        self.conv1.weight.data = torch.Tensor([[[[1.0, 1.0], [2.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]],
                                               [[[2.0, 2.0], [1.0, 1.0]], [[2.0, 1.0], [2.0, 1.0]]],
                                               [[[0.0, 5.0], [0.0, 3.0]],
                                                [[2.0, 0.0], [0.0, 10.0]]]])
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

# Update the weights
criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer.zero_grad()
target = torch.Tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Print updated weights
print(network.conv1.weight)
print(network.fc1.weight)
print(network.fc1.bias)
