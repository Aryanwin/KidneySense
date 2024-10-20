import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class ConvolutionalNeuralNet():
    def __init__(self, network):
        global device
        device = torch.device('cpu')
        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def train(self, loss_function, epochs, batch_size,
              training_set, validation_set):

        #  creating log
        log_dict = {
            'training_loss_per_epoch': [],
            'validation_loss_per_epoch': [],
            'training_accuracy_per_epoch': [],
            'validation_accuracy_per_epoch': []
        }

        #  defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)

        #  defining accuracy function
        def accuracy(network, dataloader):
            network.eval()
            total_correct = 0
            total_instances = 0
            for sample in tqdm(dataloader):
                # print(sample)
                images, labels = sample["image"], sample["label"]
                images, labels = images.to(torch.float).to(device), labels.to(torch.float).to(device)

                predictions = torch.argmax(network(images), dim=1)
                correct_predictions = sum(predictions == labels).item()
                total_correct += correct_predictions
                total_instances += len(images)
            return round(total_correct / total_instances, 3)

        #  initializing network weights
        self.network.apply(init_weights)

        #  creating dataloaders
        train_loader = DataLoader(training_set, batch_size=64,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(validation_set, batch_size=64,
                                shuffle=True, num_workers=0)

        #  setting convnet to training mode
        self.network.train()

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            train_losses = []

            #  training
            print('training...')
            for sample in tqdm(train_loader):
                #  sending data to device

                images, labels = sample["image"], sample["label"]
                # print(images)
                # print(labels)
                images, labels = images.to(torch.float).to(device), labels.to(torch.float).to(device)

                #  resetting gradients
                self.optimizer.zero_grad()

                #  making predictions
                predictions = self.network(images)

                # converting to long type
                # predictions = predictions.long()
                labels = labels.long()

                # print("predictions")
                # print(predictions.size())
                # print(predictions.type())
                # print("labels")
                # print(labels.size())
                # print(labels.type())

                #  computing loss
                loss = loss_function(predictions, labels)
                # log_dict['training_loss_per_batch'].append(loss.item())
                train_losses.append(loss.item())

                #  computing gradients
                loss.backward()

                #  updating weights
                self.optimizer.step()

            with torch.no_grad():
                print('deriving training accuracy...')

                #  computing training accuracy
                train_accuracy = accuracy(self.network, train_loader)
                log_dict['training_accuracy_per_epoch'].append(train_accuracy)

            #  validation
            print('validating...')
            val_losses = []

            #  setting convnet to evaluation mode
            self.network.eval()

            with torch.no_grad():
                for sample in tqdm(val_loader):
                    #  sending data to device
                    # print(sample)
                    images, labels = sample["image"], sample["label"]
                    images, labels = images.to(torch.float).to(device), labels.to(torch.float).to(device)

                    #  making predictions
                    predictions = self.network(images)

                    # converting to long type
                    # predictions = predictions.long()
                    labels = labels.long()
                    #  computing loss
                    # print(predictions.type())
                    # print(labels.type())
                    val_loss = loss_function(predictions, labels)
                    # log_dict['validation_loss_per_batch'].append(val_loss.item())
                    val_losses.append(val_loss.item())

                #  computing accuracy
                print('deriving validation accuracy...')
                val_accuracy = accuracy(self.network, val_loader)
                log_dict['validation_accuracy_per_epoch'].append(val_accuracy)

            train_losses = np.array(train_losses).mean()
            val_losses = np.array(val_losses).mean()

            # update epoch losses for val and train
            log_dict['training_loss_per_epoch'].append(train_losses)
            log_dict['validation_loss_per_epoch'].append(val_losses)

            print(f'training_loss: {round(train_losses, 4)}  training_accuracy: ' +
                  f'{train_accuracy}  validation_loss: {round(val_losses, 4)} ' +
                  f'validation_accuracy: {val_accuracy}\n')

        return log_dict

    def predict(self, x):
        return self.network(x)