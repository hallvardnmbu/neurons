import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.max3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 256, bias=True)
        self.fc2 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # x = self.max3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    network = Network()

    # # Print number of params
    # print("Number of parameters:", sum(p.numel() for p in network.parameters()))

    # Import CIFAR-10 dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10',
                                            train=True,
                                            download=False,
                                            transform=transform)

    # Only use 1000 samples for training
    trainset.data = trainset.data[:1000]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                shuffle=True, num_workers=2)

    # Train the network
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.1)

    for epoch in range(10):
        loss = 0.0
        for i, data in enumerate(iter(trainloader), 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = network(inputs / 255.0)
            _loss = criterion(outputs, labels)
            _loss.backward()
            optimizer.step()

            loss += _loss.item()

        # Calculate accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                outputs = network(images / 255.0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('%d loss: %.3f accuracy: %.3f' % (
        epoch, loss / len(trainloader), (correct / total) * 100))
