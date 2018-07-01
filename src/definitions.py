from os import path


DATASET_DESTINATION = path.dirname(path.abspath(__file__ + '/../')) + '/data'
TRAIN = '/LibriSpeech/train-clean-100'
TEST = '/LibriSpeech/test-clean'
PREPROCESSED_CNN_TEST = '/preprocessed/mfcc_cnn/test/'
PREPROCESSED_CNN_TRAIN = '/preprocessed/mfcc_cnn/train/'
PREPROCESSED_RNN_TEST = '/preprocessed/mfcc_rnn/test/'
PREPROCESSED_RNN_TRAIN = '/preprocessed/mfcc_rnn/train/'
LABELS_TEST = '/preprocessed/labels_test.csv'
LABELS_TRAIN = '/preprocessed/labels_train.csv'

letters = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
           'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
           's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<EMP>']
