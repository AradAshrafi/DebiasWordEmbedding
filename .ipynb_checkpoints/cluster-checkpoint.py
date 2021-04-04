from utils import extract_vectors, load_embedding, load_word_list, similarity
from sklearn.cluster import KMeans
import operator

def cluster(words, X1, random_state, y_true, num=2):
    
	kmeans_1 = KMeans(n_clusters=num, random_state=random_state).fit(X1)
	y_pred_1 = kmeans_1.predict(X1)
	correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred_1) ]
	print('precision', max(sum(correct)/float(len(correct)), 1 - sum(correct)/float(len(correct))))


# Cluster most biased words before and after debiasing

def my_cluster(wv, w2i,random_state,vocab, num_biased_words=100):
    

	male = load_word_list("data/male_words.txt",vocab)
	female = load_word_list("data/female_words.txt",vocab)

	y_true = [1]*len(male) + [0]*len(female)
	cluster(male + female, extract_vectors(male + female, wv, w2i), random_state, y_true)

def test():
	glv, glv_w2i, glv_vocab = load_embedding("data/glove.txt")
	my_cluster(glv, glv_w2i,1,glv_vocab)

	# hard_debias() # Uncomment to create the hard_debias word vector embedding
	hd_glv, hd_glv_w2i, hd_glv_vocab = load_embedding("hard_debias.txt")
	my_cluster(hd_glv, hd_glv_w2i,1,hd_glv_vocab)

def compute_word_bias(wv, w2i, vocab):
    """For each word in the vocabularly, compute bias as the difference between
    cosine similarity to the 'he' word vector and cosine similarity to the 'she'
    word vector. Return sorted list of words by bias.
    """
    
    # dictionary to hold bias for each word
    d = {}
    
    # for each word in vocab, calculate bias
    for word in vocab:
        d[word] = similarity(word, 'he', wv, w2i) - similarity(word, 'she', wv, w2i)
    
    # sort words by bias, positive bias is male leaning, negative bias female leaning
    words_sorted = sorted(gender_bias_bef.items(), key=operator.itemgetter(1))
    
    return words_sorted

def test2():
    wv, w2i, vocab = load_embedding('data/glove.txt')
    words_sorted = compute_word_bias(wv, w2i, vocab)
    print(len(words_sorted))
    print(words_sorted[:10])
    print(words_sorted[-10:])

