import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import *
from hard_debias import *

def get_components(word_vectors, num_components=20):
    """Given an array of word vectors, return the principal components

    Input: numpy array of word vectors, number of components to return
    Output: array of principal components
    """

    # create and fit principal component analysis object
    pca = PCA()
    pca.fit(word_vectors)

    # create array of principal components to return
    U = pca.components_[:num_components]

    return U

def discover_freq_dir(male_words, female_words, gender_direction,
                      principal_comps, word_vectors, word_indexes):
    """Discover the principal component that best represents the frequency
    direction by k means clustering. The direction whose removal leads to
    the worst clustering performance is selected.

    Input: array of male words, array of female words, array of candidate
        principal components
    
    Output: single principal component that represents frequency direction
    """

    S_debias = []
    for u in principal_comps:
        # new array to hold our modified male words
        male_prime = np.zeros((len(male_words), word_vectors.shape[1]))
        # remove projection on candidate principal component from each word
        for i, word in enumerate(male_words):
            # get the word vector for the word
            word_vec = word_vectors[word_indexes[word], :]
            # remove projection
            male_prime[i] = word_vec - np.dot(u, word_vec)*u

        # new array to hold our modified female words
        female_prime = np.zeros((len(female_words), word_vectors.shape[1]))
        # remove projection on candidate principal component from each word
        for i, word in enumerate(female_words):
            # get the word vector for the word
            word_vec = word_vectors[word_indexes[word], :]
            # remove projection
            female_prime[i] = word_vec - np.dot(u, word_vec)*u

        # debias the male and female word vectors
        male_hat = hard_debias2(male_prime, gender_direction)
        female_hat = hard_debias2(female_prime, gender_direction)

        # create target values for clustering precision
        # male words are target = 1, female are target 0
        male_targets = np.ones(male_hat.shape[0])
        female_targets = np.zeros(female_hat.shape[0])
        # concatenate to get all targets
        targets = np.concatenate((male_targets, female_targets), axis=0)
        # concatenate male and female word vectors for clustering
        combined_vecs = np.concatenate((male_hat, female_hat), axis=0)

        # create k means cluster object and run test
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(combined_vecs)
        predictions = kmeans.predict(combined_vecs)
        correct = [1 if a == b else 0 for (a, b) in zip(targets, predictions)]
        precision = max(sum(correct)/float(len(correct)), 1 - sum(correct)/float(len(correct)))
        S_debias.append(precision)
        print(precision)

    # the frequency principal component is the one that corresponds to the worst
    # performance on the clustering test
    i = np.argmin(S_debias)
    return i, principal_comps[i]

def double_hard_debias(embedding_filepath, male_word_filepath, 
                       female_word_filepath, gender_pair_filepath):
    """Performs the double-hard debias algorithm on a given word embedding.

    Input: filepath to word embedding,
        filepath to list of male specific words, 
        filepath to female specific words

    Output: double-hard debiased word embedding
    """
    
    print('Loading word files...')

    # load word embedding and gender specific words
    word_vectors, word_indexes, vocab = load_embedding(embedding_filepath)
    male_words = load_word_list(male_word_filepath, vocab)
    female_words = load_word_list(female_word_filepath, vocab)
    set_of_pairs = load_def_pairs(gender_pair_filepath)

    print('Word files loaded')

    # decentralize and normalize the word vectors
    word_vectors = decentralize(word_vectors)
    word_vectors = normalize(word_vectors)

    print('Word vectors decentralized and normalized')

    # compute principal components
    U = get_components(word_vectors)

    print('Principal components computed')
    
    # find gender direction
    mu_list = calculate_mu(set_of_pairs, word_vectors, word_indexes)
    gender_subspace = calculate_gender_direction(set_of_pairs, mu_list, word_vectors, word_indexes, num_components=1)
    gender_direction = gender_subspace[0]
    
    print('Gender direction found')

    # discover the frequency direction
    i, u = discover_freq_dir(male_words, female_words, gender_direction, U, word_vectors, word_indexes)

    print('Frequency direction discovered, U[{}]'.format(i))

    # remove component on frequency direction
    word_vectors_prime = np.zeros(word_vectors.shape)
    for j in range(word_vectors.shape[0]):
        word_vectors_prime[j] = word_vectors[j] - np.dot(word_vectors[j], u)*u

    print('Frequency direction removed')
    
    # remove component on gender direction
    word_vectors_debiased = hard_debias2(word_vectors_prime, gender_direction)

    print('Gender direction removed')
    
    word_vectors_debiased = normalize(word_vectors_debiased)
    
    print('Word vectors normalized')
    
    
    
    print('Complete')

    return word_vectors_debiased