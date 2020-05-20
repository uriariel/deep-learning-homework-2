from time import time

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from config import Config
from nets import Net

SHOW_SAMPLES = True
EPOCHS = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def dataset_split_rule(x_set):
    return [indx for indx, target_class in enumerate(x_set.targets) if target_class in range(0, 6 + 1)]


trainset = datasets.MNIST(Config.training_data_path, download=False, train=True, transform=transform)
valset = datasets.MNIST(Config.test_data_path, download=False, train=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, sampler=SubsetRandomSampler(dataset_split_rule(x_set=trainset)))
valloader = DataLoader(valset, batch_size=64, sampler=SubsetRandomSampler(dataset_split_rule(x_set=valset)))

# images, labels = list(trainloader())
#
# print(images.shape)
# print(labels.shape)
#
# if SHOW_SAMPLES:
#     plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
#
#     figure = plt.figure()
#     num_of_images = 60
#     for index in range(1, num_of_images + 1):
#         plt.subplot(6, 10, index)
#         plt.axis('off')
#         plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

# image size is 28x28
torch.manual_seed(42)
model = Net(D=1, H1=10, H2=20, class_count=10).cuda()
model.train()

criterion = nn.NLLLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
optimizer = optim.Adam(model.parameters())

time0 = time()
for e in range(EPOCHS):
    running_loss = 0
    for images, labels in trainloader:
        images = images.cuda()
        labels = labels.cuda()
        # Flatten MNIST images into a 784 long vector
        # images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

print("\nTraining Time (in minutes) =", (time() - time0) / 60)

torch.save(model, 'zero_to_six.pt')

model.eval()

acc, count = 0, 0
with torch.no_grad():
    for images, labels in valloader:
        images = images.cuda()
        labels = labels.cuda()

        logps = model(images)
        ps = torch.exp(logps)
        pred_labels = torch.argmax(ps, 1)
        acc += torch.sum(pred_labels == labels).item() / len(labels)
        count += 1

print("\nModel Accuracy =", acc / count)
