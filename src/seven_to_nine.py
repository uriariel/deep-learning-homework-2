from time import time

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from config import Config
from nets import Net

SHOW_SAMPLES = True


def dataset_split_rule(x_set):
    return [indx for indx, target_class in enumerate(x_set.targets) if target_class in range(7, 9 + 1)]


def net_from_transfer():
    model = torch.load('zero_to_six.pt')
    model.conv1.weight.requires_grad = False
    model.conv1.bias.requires_grad = False
    model.conv2.weight.requires_grad = False
    model.conv2.bias.requires_grad = False
    model.fc2 = nn.Linear(50, 10).cuda()
    return model


def empty_net():
    return Net(D=1, H1=10, H2=20, class_count=10).cuda()


def train_model(train_data, model, optimizer,criterion = nn.NLLLoss(), epochs = 10):
    model.train()
    time0 = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_data:
            # Flatten MNIST images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            images = images.cuda()
            labels = labels.cuda()
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
            print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_data)))

    print("\nTraining Time (in minutes) =", (time() - time0) / 60)

#torch.save(model, 'zero_to_six.pt')

def evaluate_model(test_data, model):
    model.eval()
    acc, count = 0, 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.cuda()
            labels = labels.cuda()
            logps = model(images)
            ps = torch.exp(logps)
            pred_labels = torch.argmax(ps, 1)
            acc += torch.sum(pred_labels == labels).item() / len(labels)
            count += 1

    print("\nModel Accuracy =", acc / count)

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

criterion = nn.NLLLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

print("Question 3 learning the 0-6 model".center(100, '-'))
model = net_from_transfer()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
train_model(model,optimizer, trainloader)
torch.save(model, 'zero_to_six.pt')
evaluate_model(model, valloader)

print()
print("Question 4 test 7-9  - learning from 0-6 model".center(100, '-'))
model = net_from_transfer()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
train_model(model,optimizer, trainloader)
evaluate_model(model, valloader)

print()
print("Question 5 test 7-9  - learning from scratch".center(100, '-'))
model = empty_net()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
train_model(trainloader, model,optimizer)
evaluate_model(valloader, model)


#torch.save(model, 'zero_to_six.pt')

