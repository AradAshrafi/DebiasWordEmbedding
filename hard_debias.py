import numpy as np
from sklearn.decomposition import PCA
import json
from utils import load_embedding


def calculate_mu(set_of_pairs, word_vectors, word_indices):
    """
    It will return Mu for each of the male-female pairs
    """
    mu_list = []
    for D in set_of_pairs:
        mu_i = (word_vectors[word_indices[D[0].lower()]] + word_vectors[word_indices[D[1].lower()]]) / 2
        mu_list.append(mu_i)

    return mu_list


def calculate_gender_direction(set_of_pairs, mu_list, word_vectors, word_indices, num_components=1):
    """
    We'll return the top components of Principal Component of the C matrix as the gender direction
    num_components=1 ====> gender direction
    """
    C_matrix = np.zeros([300, 300])
    for i in range(len(set_of_pairs)):
        C_matrix += np.matmul(np.asarray([word_vectors[word_indices[set_of_pairs[i][0].lower()]] - mu_list[i]]).T,
                              np.asarray([word_vectors[word_indices[set_of_pairs[i][0].lower()]] - mu_list[i]])) / 2
        C_matrix += np.matmul(np.asarray([word_vectors[word_indices[set_of_pairs[i][1].lower()]] - mu_list[i]]).T,
                              np.asarray([word_vectors[word_indices[set_of_pairs[i][1].lower()]] - mu_list[i]])) / 2

    pca = PCA()
    pca.fit(C_matrix)

    # When num_components==1 => gender_subspace => gender vector
    gender_subspace = pca.components_[:num_components]

    return gender_subspace


def hard_debias(path_to_embedding="Double-Hard Debias/embeddings/glove.txt",
                path_to_def_pairs="Hard Debias/Data/definitional_pairs.json"):
    word_vectors, word_indices, _ = load_embedding(path_to_embedding)
    word_vectors = np.asarray(word_vectors)

    with open(path_to_def_pairs) as f:
        set_of_pairs = json.load(f)

    mu_list = calculate_mu(set_of_pairs, word_vectors, word_indices)
    gender_subspace = calculate_gender_direction(set_of_pairs, mu_list, word_vectors, word_indices, num_components=1)
    gender_direction = gender_subspace[0]

    ### Subtracting Gender Bias from each Word Vector
    for i in range(len(word_vectors)):
        word_vectors[i] = word_vectors[i] - np.dot(word_vectors[i], gender_direction) * gender_direction

    return word_vectors  # Hard Debiased embeddings
