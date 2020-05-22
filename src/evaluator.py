import torch
import numpy as np
import matplotlib.pyplot as plt


class NetworkEvaluator:
    def __init__(self, config, network):
        self.set_constant_seed()
        self.config = config
        self.network = network

    def run(self, data):
        # Config objects
        num_of_epochs = self.config.NUM_EPOCHS
        criterion = self.config.CRITERION
        activation_func = self.config.ACTIVATION_FUNC
        optimizer = self.config.OPTIMIZER
        learning_rate = self.config.LR

        print(
            f' Evaluating for: {self.network}, activation func: {activation_func.__name__} and optimizer: {optimizer.__name__} '.center(
                100, '-'))
        self.evaluate_network(self.network(), data, criterion, optimizer, num_of_epochs, learning_rate)

    def set_constant_seed(self):
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)

    def evaluate_network(self, net, data, criterion, optimizer, num_of_epochs, learning_rate):
        X_train, Y_train, X_test, Y_test = data

        train_losses = []
        test_losses = []

        # train network
        for e in range(num_of_epochs):
            train_losses += [
                self.train_epoch(model=net, opt=optimizer(net.parameters(), lr=learning_rate), criterion=criterion,
                                 X=X_train,
                                 Y=Y_train)]
            test_losses += [criterion(net(X_test), Y_test)]

        net.eval()

        # run test set
        out = net(X_test)
        acc = self.get_accuracy(out, Y_test)

        print(f'parameters count is {self.get_vector_parameters_count(net)}')
        print(f'vector parameters count is {self.get_parameters_count(net)}')
        print(f'Number of Epochs: {num_of_epochs}.')
        print(f'Model accuracy: {acc}%')

        self.plot_losses(torch.linspace(1, num_of_epochs, num_of_epochs), train_losses, test_losses)

    def train_epoch(self, model, opt, criterion, X, Y):
        model.train()

        # (0) Zero gradient for computation
        opt.zero_grad()
        # (1) Forward
        y_out = model(X)
        # (2) Compute diff
        loss = criterion(y_out, Y)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()

        return loss

    def train_epoch_manually(self, model, opt, criterion, X, Y):
        model.train()

        model.zero_grad()
        # (1) Forward
        y_out = model(X)
        # (2) Compute diff
        loss = criterion(y_out, Y)
        # (3) Compute gradients
        loss.backward()

        # (4) update weights
        for p in model.parameters():
            p.data -= p.grad.data * self.config.LR

        return loss

    def get_parameters_count(self, model):
        return sum(self.torch_len(p) for p in model.parameters())

    @staticmethod
    def torch_len(tensor):
        return tensor.size()[0]

    @staticmethod
    def get_vector_parameters_count(model):
        return len(list(model.parameters()))

    @staticmethod
    def plot_losses(epochs, train_loss, test_loss, title = ''):
        plt.plot(epochs, train_loss, 'r')
        plt.plot(epochs, test_loss, 'b')
        plt.title(title + ' red - train loss, blue - test loss')
        plt.show()

    @staticmethod
    def get_accuracy(y_pred, y_test):
        pred = torch.round(y_pred).detach().numpy()

        # convert ground truth to numpy
        ynp = y_test.data.numpy()

        return 100 * (np.count_nonzero(ynp == pred) / len(pred))



