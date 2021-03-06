import torch
import torch.optim as optim
import ctcdecode
import multiprocessing
import numpy as np

from src.rnn_ctc.dataset_loader import DatasetLoader
from src.rnn_ctc.rnn_module import RNNModule
from warpctc_pytorch import CTCLoss

from src.definitions import letters
from src.definitions import MODEL_DESTINATION
from src.definitions import DATASET_DESTINATION
from src.definitions import LABELS_TRAIN
from src.definitions import PREPROCESSED_RNN_TRAIN

from src.encoding import decode, make_encodings

RNN_WEIGHTS = '/rnn_weights.pth'


class RNNTrainer():
    def __init__(self, args):
        self.num_epoches = args['num_epoches']
        self.data_loader = DatasetLoader(args['labels_path'],
                                         args['features_path'])
        self.input_size = args['input_size']
        self.num_layers = args['num_layers']
        self.hidden_units = args['hidden_units']
        self.vocabulary_size = args['vocab_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        # init model
        model = RNNModule(self.input_size,
                          self.num_layers,
                          self.hidden_units,
                          self.vocabulary_size).to(self.device)

        # define ctc loss function
        ctc_loss = CTCLoss()
        # define optimizer
        adam = optim.Adam(model.parameters())

        print("Starting training..")
        for epoch in range(self.num_epoches):
            loss_acc = 0
            index_array = np.arange(len(self.data_loader))
            np.random.shuffle(index_array)
            for index, sample in enumerate(index_array):
                adam.zero_grad()
                features, labels = self.data_loader[sample]
                labels_size = torch.IntTensor([labels.shape[0]])
                features_size = torch.IntTensor([features.shape[0]])

                # place data on gpu
                features.to(self.device)
                labels.to(self.device)

                output = model(features, self.device)
                loss = ctc_loss(output, labels, features_size, labels_size)

                loss.backward()
                adam.step()
                # accumulate loss
                loss_acc += loss.data[0]
                if (index + 1) % 100 == 0:
                    print("Training example {} -- Loss: {:.4f}".format(index + 1, loss_acc/(index + 1)))

                if(index % 2000) == 0:
                    torch.save(model.state_dict(), MODEL_DESTINATION + RNN_WEIGHTS)


            print("Epoch: {} ---> Mean loss: {:.4f}".format(epoch + 1, loss_acc/self.data_loader.__len__()))
            torch.save(model.state_dict(), MODEL_DESTINATION + RNN_WEIGHTS)
        print("Training finished..")
        return model

    def evaluate(self, model):
        pass

    def inference(self, model, output):
        pass

    def greedy_decoding(self, probs, letter2ind):
        indices = []
        letter2ind = make_encodings()
        for i in range(probs.shape[0]):
            indices.append(np.argmax(probs[i, :]))

        print("True: {}".format(decode(label, letter2ind)))
        print("Predicted: {}".format(decode(indices, letter2ind)))


if __name__ == "__main__":
    args = {
        "num_epoches": 10,
        "labels_path": DATASET_DESTINATION + LABELS_TRAIN,
        "features_path": DATASET_DESTINATION + PREPROCESSED_RNN_TRAIN,
        "vocab_size": 28,
        "input_size": 26,
        "num_layers": 1,
        "hidden_units": 512
        }

    trainer = RNNTrainer(args)
    # trainer.train()
    model = RNNModule(trainer.input_size,
                      trainer.num_layers,
                      trainer.hidden_units,
                      trainer.vocabulary_size).to(trainer.device)
    softmax = torch.nn.Softmax(dim=2)
    model.load_state_dict(torch.load(MODEL_DESTINATION + RNN_WEIGHTS))
    model.eval()
    mfcc, label = trainer.data_loader[0]
    probs = model(mfcc, trainer.device)
    probs = softmax(probs)
    probs = probs.squeeze(1).detach().cpu().numpy()

    # greedy decoding
    trainer.greedy_decoding(probs, letter2ind)
