from time import time

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from config import Config
from nets import Net

EPOCHS = 15
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def plot_losses(epochs, train_loss, test_loss, title=''):
    plt.style.use("ggplot")
    plt.plot(epochs, train_loss, 'r', label="Training Loss")
    plt.plot(epochs, test_loss, 'b', label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="center")
    plt.title(title)
    plt.show()


def dataset_split_rule(x_set, min, max):
    return [indx for indx, target_class in enumerate(x_set.targets) if target_class in range(min, max + 1)]


def transferred_network(model_path):
    model = torch.load(model_path)
    model.conv1.weight.requires_grad = False
    model.conv1.bias.requires_grad = False
    model.conv2.weight.requires_grad = False
    model.conv2.bias.requires_grad = False
    model.fc2 = nn.Linear(50, 10).cuda()
    return model


def empty_network():
    return Net(D=1, H1=10, H2=20, class_count=10).cuda()


def train_model(train_data, test_data, model, optimizer, criterion=nn.NLLLoss(), epochs=EPOCHS, title=''):
    model.train()
    time0 = time()
    train_losses = []
    test_losses = []

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
            train_losses += [(running_loss / len(train_data))]
            test_losses += [sum([criterion(model(images.cuda()), test_labels.cuda()).item() for images, test_labels in
                                 test_data]) / len(test_data)]
            print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_data)))

    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    plot_losses(torch.linspace(1, epochs, epochs), train_losses, test_losses, title)


def evaluate_model(test_data, model):
    model.eval()
    acc, count = 0, 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.cuda()
            labels = labels.cuda()

            pred_labels = torch.argmax(torch.exp(model(images)), 1)
            acc += torch.sum(pred_labels == labels).item() / len(labels)
            count += 1

    print("\nModel Accuracy =", acc / count)


def main():
    trainset = datasets.MNIST(Config.training_data_path, download=False, train=True, transform=transform)
    valset = datasets.MNIST(Config.test_data_path, download=False, train=False, transform=transform)

    train_data_zero_to_six = DataLoader(trainset, batch_size=64,
                                        sampler=SubsetRandomSampler(dataset_split_rule(x_set=trainset, min=0, max=6)))
    test_data_zero_to_six = DataLoader(valset, batch_size=64,
                                       sampler=SubsetRandomSampler(dataset_split_rule(x_set=valset, min=0, max=6)))
    train_data_seven_to_nine = DataLoader(trainset, batch_size=64,
                                          sampler=SubsetRandomSampler(dataset_split_rule(x_set=trainset, min=7, max=9)))
    test_data_seven_to_nine = DataLoader(valset, batch_size=64,
                                         sampler=SubsetRandomSampler(dataset_split_rule(x_set=valset, min=7, max=9)))

    torch.manual_seed(42)

    print("learning 0-6 model".center(100, '-'))
    model = empty_network()
    optimizer = optim.Adam(model.parameters())

    train_model(train_data_zero_to_six,
                test_data_zero_to_six,
                model,
                optimizer,
                title="0-6 model")

    torch.save(model, 'zero_to_six.pt')
    evaluate_model(test_data_zero_to_six, model)

    print("learning 7-9 based on 0-6 model".center(100, '-'))
    model = transferred_network('zero_to_six.pt')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    train_model(train_data_seven_to_nine,
                test_data_seven_to_nine,
                model,
                optimizer,
                title="7-9 based on 0-6 model")

    evaluate_model(test_data_seven_to_nine, model)

    print()
    print("learning 7-9 model from scratch".center(100, '-'))
    model = empty_network()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    train_model(train_data_seven_to_nine,
                test_data_seven_to_nine,
                model,
                optimizer,
                title="7-9 model from scratch")

    evaluate_model(test_data_seven_to_nine, model)


if __name__ == '__main__':
    main()
