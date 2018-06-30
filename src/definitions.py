from os import path


DATASET_DESTINATION = path.dirname(path.abspath(__file__ + '/../')) + '/data'
TRAIN = '/LibriSpeech/train-clean-100'
TEST = '/LibriSpeech/test-clean'
PREPROCESSED_TEST = '/preprocessed/mfcc/test/'
PREPROCESSED_TRAIN = '/preprocessed/mfcc/train/'
LABELS_TEST = '/preprocessed/labels_test.csv'
LABELS_TRAIN = '/preprocessed/labels_train.csv'

letters = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
           'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
           's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<EMP>']
