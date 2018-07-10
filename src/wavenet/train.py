import torch
from definitions import *
from wavenet.data_loader import SpeechDataLoader
from wavenet.wavenet import Wavenet
from warpctc_pytorch import CTCLoss
from definitions import letters


class Trainer():
    def __init__(self, args):
        self.args = args
        self.epochs_num = args['epochs_num']

        self.data_loader = SpeechDataLoader(args['train_labels_path'], args['train_files_path'], 
                                            args['batch_size'])

        self.model = Wavenet(args['input_channels'], args['stack_len'],
                             args['dilations_per_layer'], args['res_layers_num'],
                             args['softmax_output']).cuda()
        self.loss = CTCLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def run(self):
        for epoch in range(0, self.epochs_num):
            number_of_samples = 0
            loss_sum = 0
            for i, batch in enumerate(self.data_loader):
                inputs, targets, input_percentages, target_sizes = batch
                inputs = inputs.cuda()

                outputs = self.model(inputs)
                # output is empty tensor if length of inputs is too small
                if outputs.size(0) > 0:
                    number_of_samples += 1
                    outputs = outputs.transpose(0, 1)

                    seq_length = outputs.size(0)
                    sizes = input_percentages.mul_(int(seq_length)).int()
                    loss = self.loss(outputs, targets, sizes, target_sizes)

                    loss = loss / inputs.size(0)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_sum += loss.item()
                    if number_of_samples % 100 == 0:
                        print('{}: Loss: {}'.format(number_of_samples, loss_sum / number_of_samples))
            
            average_loss = loss_sum / number_of_samples
            print('\n\n===> Average loss: {}'.format(average_loss))


if __name__ == '__main__':
    args = {
        'train_labels_path': DATASET_DESTINATION + LABELS_TRAIN,
        'train_files_path': DATASET_DESTINATION + PREPROCESSED_CNN_TRAIN,
        'batch_size': 1,
        'epochs_num': 10,
        'input_channels': 20,
        'stack_len': 5,
        'dilations_per_layer': 5,
        'res_layers_num': 512,
        'softmax_output': len(letters)
    }
    trainer = Trainer(args)
    trainer.run()
