import numpy as np

def load_embedding(filepath):
    """Load word vector embedding

    input: path to file containing embeddings
    output: numpy array of word vectors, dictionary of words to index in array,
        list of words ie. vocabulary
    """

    # open file and read each all lines
    embedding_file = open(filepath)
    lines = embedding_file.readlines()

    word_vectors = []
    vocab = []

    # iterate through each line, add word to vocab and add vector to word_vectors
    for l in lines:
        # split line on whitespace
        tokens = l.strip().split(" ")
        vocab.append(tokens[0])
        word_vectors.append([float(token) for token in tokens[1:]])

    # create dictionary to allow us to quickly get vocab index for any word
    word_indexes = {word: idx for idx, word in enumerate(vocab)}

    # create numpy array for word vectors for easier processing
    word_vectors = np.array(word_vectors)

    return word_vectors, word_indexes, vocab