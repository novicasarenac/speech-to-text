from definitions import *
from wavenet.data_loader import SpeechDataLoader


class Trainer():
    def __init__(self, args):
        self.args = args
        self.epochs_num = args['epochs_num']

        self.data_loader = SpeechDataLoader(args['labels_path'], args['files_path'], 
            args['batch_size'])
    
    def run(self):
        for epoch in range(0, self.epochs_num):
            for i, batch in enumerate(self.data_loader):
                print(i)
                inputs, targets, input_percentages, target_sizes = batch
                print('Inputs size: {}, target size: {}'.format(inputs.size(), targets.size()))


if __name__ == '__main__':
    args = {
        'labels_path': DATASET_DESTINATION + LABELS_TEST,
        'files_path': DATASET_DESTINATION + PREPROCESSED_TEST,
        'batch_size': 1,
        'epochs_num': 1
    }
    trainer = Trainer(args)
    trainer.run()
