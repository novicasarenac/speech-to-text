import string
from src.definitions import letters


def make_encodings():
    letter2ind = {}
    for i, letter in enumerate(letters):
        letter2ind[letter] = i
    return letter2ind


def encode(text, letter2ind):
    clean_text = text.lower()
    translator = clean_text.maketrans('', '', string.punctuation)
    clean_text = clean_text.translate(translator)
    encoded_text = []
    for character in clean_text:
        encoded_text.append(letter2ind[character])
    return encoded_text


def decode(encoded_letters, letter2ind):
    sentence = []
    for letter_ind in encoded_letters:
        if letter_ind != 0:
            sentence.append(find_char_by_index(letter_ind, letter2ind))

    return ''.join(sentence)


def find_char_by_index(letter_index, letter2ind):
    for key, value in letter2ind.items():
        if value == letter_index:
                return key

