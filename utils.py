import numpy as np
import json
from scipy.spatial import distance 
import itertools

WEAT_words = {
'A':['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill'], 
'B':['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna'],
'C':['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career'],
'D':['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives'],
'E':['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition'],
'F':['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture'],
'G':['science', 'technology', 'physics', 'chemistry', 'einstein', 'nasa', 'experiment', 'astronomy'],
'H':['poetry', 'art', 'shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama'],
}

def similarity(w1, w2, wv, w2i):
    
    i1 = w2i[w1]
    i2 = w2i[w2]
    vec1 = wv[i1, :]
    vec2 = wv[i2, :]

    return 1-(distance.cosine(vec1, vec2))

def association_diff(t, A, B, wv, w2i):
    
    mean_a = []
    mean_b = []
    
    for a in A:
        mean_a.append(similarity(t, a, wv, w2i))
    for b in B:
        mean_b.append(similarity(t, b, wv, w2i))
        
    mean_a = sum(mean_a)/float(len(mean_a))
    mean_b = sum(mean_b)/float(len(mean_b))
    
    return mean_a - mean_b

def effect_size(X, Y, A, B,  wv, w2i, vocab):
    
    assert(len(X) == len(Y))
    assert(len(A) == len(B))
    
    norm_x = []
    norm_y = []
    
    for x in X:
        norm_x.append(association_diff(x, A, B, wv, w2i))
    for y in Y:
        norm_y.append(association_diff(y, A, B, wv, w2i))
    
    std = np.std(norm_x+norm_y, ddof=1)
    norm_x = sum(norm_x) / float(len(norm_x))
    norm_y = sum(norm_y) / float(len(norm_y))
    
    return (norm_x-norm_y)/std

def s_word(w, A, B, wv, w2i, vocab, all_s_words):
    
    if w in all_s_words:
        return all_s_words[w]
    
    mean_a = []
    mean_b = []
    
    for a in A:
        mean_a.append(similarity(w, a, wv, w2i))
    for b in B:
        mean_b.append(similarity(w, b, wv, w2i))
        
    mean_a = sum(mean_a)/float(len(mean_a))
    mean_b = sum(mean_b)/float(len(mean_b))
    
    all_s_words[w] = mean_a - mean_b

    return all_s_words[w]


def s_group(X, Y, A, B,  wv, w2i, vocab, all_s_words):
    
    total = 0
    for x in X:
        total += s_word(x, A, B,  wv, w2i, vocab, all_s_words)
    for y in Y:
        total -= s_word(y, A, B,  wv, w2i, vocab, all_s_words)
        
    return total

def p_value_test(X, Y, A, B,  wv, w2i, vocab):
    
    if len(X) > 10:
        print('might take too long, use sampled version: p_value')
        return
    
    assert(len(X) == len(Y))
    
    all_s_words = {}
    s_orig = s_group(X, Y, A, B, wv, w2i, vocab, all_s_words) 
    
    union = set(X+Y)
    subset_size = int(len(union)/2)
    
    larger = 0
    total = 0
    for subset in set(itertools.combinations(union, subset_size)):
        total += 1
        Xi = list(set(subset))
        Yi = list(union - set(subset))
        if s_group(Xi, Yi, A, B, wv, w2i, vocab, all_s_words) > s_orig:
            larger += 1
    print('num of samples', total)
    return larger/float(total)

def extract_vectors(words, wv, w2i):
    
    X = [wv[w2i[x],:] for x in words]
    return X

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

def load_def_pairs(filepath):
    """Load gendered word pairs from json file
    """
    
    with open(filepath) as f:
        set_of_pairs = json.load(f)
    
    return set_of_pairs