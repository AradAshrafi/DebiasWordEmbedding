from utils import extract_vectors, load_embedding, load_word_list
from sklearn.cluster import KMeans

def cluster(words, X1, random_state, y_true, num=2):
    
	kmeans_1 = KMeans(n_clusters=num, random_state=random_state).fit(X1)
	y_pred_1 = kmeans_1.predict(X1)
	correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred_1) ]
	print('precision', max(sum(correct)/float(len(correct)), 1 - sum(correct)/float(len(correct))))


# Cluster most biased words before and after debiasing

def my_cluster(wv, w2i,random_state,vocab):

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

test()








