from utils import extract_vectors, load_embedding, similarity, simi
from sklearn.cluster import KMeans
import operator
from hard_debias import hard_debias
from double_hard_debias import double_hard_debias

def cluster(words, X1, random_state, y_true, num=2):
    
	kmeans_1 = KMeans(n_clusters=num, random_state=random_state).fit(X1)
	y_pred_1 = kmeans_1.predict(X1)
	correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred_1) ]
	print('precision', max(sum(correct)/float(len(correct)), 1 - sum(correct)/float(len(correct))))


# Cluster most biased words before and after debiasing

def my_cluster(wv, w2i,random_state,vocab, words_sorted, num_biased_words=100):
    

	male = [pair[0] for pair in words_sorted[-num_biased_words:]]
	female = [pair[0] for pair in words_sorted[:num_biased_words]]

	y_true = [1]*len(male) + [0]*len(female)
	cluster(male + female, extract_vectors(male + female, wv, w2i), random_state, y_true)


def compute_word_bias(wv, w2i, vocab, he, she):
    """For each word in the vocabularly, compute bias as the difference between
    cosine similarity to the 'he' word vector and cosine similarity to the 'she'
    word vector. Return sorted list of words by bias.
    """
    
    # dictionary to hold bias for each word
    d = {}
    
    # for each word in vocab, calculate bias
    for word in vocab:
        v = wv[w2i[word], :]
        d[word] = simi(v, he) - simi(v, she)
    
    # sort words by bias, positive bias is male leaning, negative bias female leaning
    words_sorted = sorted(d.items(), key=operator.itemgetter(1))
    
    return words_sorted


def test():
    glv, glv_w2i, glv_vocab = load_embedding("data/glove.txt")
    words_sorted = compute_word_bias(glv, glv_w2i, glv_vocab)
    for n in [100, 500, 1000]:
        my_cluster(glv, glv_w2i, 1, glv_vocab, words_sorted, n)

    hard_debias() # Uncomment to create the hard_debias word vector embedding
    hd_glv, hd_glv_w2i, hd_glv_vocab = load_embedding("hard_debias.txt")
    
    for n in [100, 500, 1000]:
        my_cluster(hd_glv, hd_glv_w2i, 1, hd_glv_vocab, words_sorted, n)
        
    embedding_filepath = './data/glove.txt'
    male_filepath = './data/male_words.txt'
    female_filepath = './data/female_words.txt'
    pairs_filepath = './data/definitional_pairs.json'
    
    dbl_glv, dbl_w2i, dbl_vocab = double_hard_debias(embedding_filepath, male_filepath, female_filepath, pairs_filepath)
    
    for n in [100, 500, 1000]:
        my_cluster(dbl_glv, dbl_w2i, 1, dbl_vocab, words_sorted)

#test()