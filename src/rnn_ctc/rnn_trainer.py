import torch
import torch.optim as optim

from src.rnn_ctc.dataset_loader import DatasetLoader
from src.rnn_ctc.rnn_module import RNNModule
from warpctc_pytorch import CTCLoss
from src.definitions import DATASET_DESTINATION
from src.definitions import LABELS_TRAIN
from src.definitions import PREPROCESSED_RNN_TRAIN


class RNNTrainer():
    def __init__(self, args):
        self.num_epoches = args['num_epoches']
        self.data_loader = DatasetLoader(args['labels_path'],
                                         args['features_path'])
        self.input_size = args['input_size']
        self.num_layers = args['num_layers']
        self.hidden_units = args['hidden_units']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        # init model
        model = RNNModule(self.input_size, self.num_layers, self.hidden_units)
        model.to(self.device)

        # define ctc loss function
        ctc_loss = CTCLoss(length_average=True)
        # define optimizer
        adam = optim.Adam(model.parameters())

        print("Starting training..")
        for epoch in self.num_epoches:
            loss_acc = 0
            for sample in self.data_loader.__len__():
                adam.zero_grads()
                features, labels = self.data_loader.__getitem__(sample)
                labels_size = torch.IntTensor([labels.shape[0]])
                features_size = torch.IntTensor([features.shape[2]])

                # place data on gpu
                features.to(self.device)
                labels.to(self.device)

                # compute gradients with autograd
                features.require_grad(True)

                output = model(features, self.device)
                loss = ctc_loss(output, labels, features_size, labels_size)

                loss.backward()
                adam.step()
                # accumulate loss
                loss_acc += loss
                if sample % 100 == 0:
                    print("Training example {} -- Loss: {:.4f}".format(sample + 1, loss_acc/(sample + 1)))

            print("Epoch: {} ---> Mean loss: {:.4f}".format(epoch + 1, loss_acc/self.data_loader.__len__()))
        print("Training finished..")


if __name__ == "__main__":
    args = {
        "num_epoches": 1,
        "labels_path": DATASET_DESTINATION + LABELS_TRAIN,
        "features_path": DATASET_DESTINATION + PREPROCESSED_RNN_TRAIN,
        "input_size": 13,
        "num_layers": 1,
        "hidden_units": 512
        }

    trainer = RNNTrainer(args)
    trainer.train()
