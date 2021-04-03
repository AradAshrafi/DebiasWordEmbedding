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


def recreate_embedding(word_vectors, vocab, new_embedding_name="hard_debais"):
  with open(new_embedding_name + ".txt", "w") as text_file:
    for i in range(len(vocab)):
      text_file.write(vocab[i]+ " ")
      text_file.write(' '.join([str(element) for element in word_vectors[i]]) + '\n')


def normalize(word_vectors):
    """Given an array of word vectors, return an array of those vectors
    normalized to l2 norm = 1

    Input: m by n array of word vectors

    Output: m by n array of word vectors
    """

    # get norm for each row in word vector matrix
    norms = np.apply_along_axis(np.linalg.norm, 1, word_vectors)
    norms = norms.reshape((norms.size, 1))

    # create new matrix of normalized word vectors
    normalized_word_vectors = word_vectors / norms

    return normalized_word_vectors

def decentralize(word_vectors):
    """Given an array of word vectors, return an array of those
    vectors decentralized, ie. subtract the average vector from
    each vector.

    Input: m by n array of word vectors

    Output: m by n array of word vectors
    """

    # calculate average vector for array of word vectors
    mean_vector = np.mean(word_vectors, axis=0)

    # create decentralized word vectors by subtracting mean from each vector
    decentralized_word_vectors = word_vectors - mean_vector

    return decentralized_word_vectors

def load_word_list(filepath, vocab):
    """Create array of words, usually some subset of the total vocabulary
    that is needed for identifying gender or testing bias.

    Input: filepath to file with one word per line

    Output: array of strings
    """

    # read in and clean all words from file
    word_file = open(filepath)
    words = word_file.readlines()
    words = [word.strip() for word in words]

    # only keep words that exist in our vocabulary
    kept_words = []
    for word in words:
        if word in vocab:
            kept_words.append(word)

    return kept_words