import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*2, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = F.softmax(self.fc1(x), dim=1)

        return x


if __name__ == '__main__':
    network = Network()

    # Import CIFAR-10 dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # Downasmple to 14x14
        torchvision.transforms.Resize((14, 14))
    ])

    trainset = torchvision.datasets.MNIST(root='./datasets/mnist',
                                          train=True,
                                          download=False,
                                          transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                shuffle=True, num_workers=2)

    # Train the network
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1, momentum=0)

    losses = []
    for epoch in range(25):
        loss = 0.0
        for i, data in enumerate(iter(trainloader), 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = network(inputs / 255.0)
            _loss = criterion(outputs, labels)
            _loss.backward()
            optimizer.step()

            loss += _loss.item()
        losses.append(loss / len(trainloader))

        if epoch > 1 and epoch % 5 == 0:

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

            print('%d loss: %.3f accuracy: %.3f' % (epoch, sum(losses[epoch-5:epoch]) / 5, correct / total))
