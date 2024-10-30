import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4, 256, bias=True)
        self.fc2 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = F.relu(self.conv3(x))
        x = self.max3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    network = Network()

    # Set all weights to 1.0
    with torch.no_grad():
        for param in network.parameters():
            param.fill_(1.0)

    # Print number of params
    print("Number of parameters:", sum(p.numel() for p in network.parameters()))

    # Define the input as a 3x32x32 tensor of 0.1s.
    input = torch.ones(1, 3, 32, 32) * 0.1
    target = torch.tensor([4])

    # Train the network.
    softmax = nn.Softmax(dim=1)
    criterion = lambda x, target: F.cross_entropy(softmax(x), target)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    loss = 0
    for epoch in range(5):
        print(epoch)
        optimizer.zero_grad()
        outputs = network(input)

        _loss = criterion(outputs, target)
        _loss.backward()

        # Print gradients before update
        for name, param in network.named_parameters():
            if param.grad is not None:
                print(f"{name} gradients:")
                print(param.grad[0])
                print("----")

        optimizer.step()
        loss += _loss.item()

    import numpy as np
    # Print all weights
    for param in network.parameters():
        print(np.unique(param.detach().numpy().flatten()))
        # np.set_printoptions(threshold=np.inf)
        # print(param.detach().numpy())
