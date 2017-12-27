from __future__ import division
import ot, numpy as np, sys, codecs, string, editdistance
from scipy import sparse
from sklearn.metrics import euclidean_distances
from sklearn.externals.joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
from sklearn.metrics.scorer import check_scoring
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathos.multiprocessing import ProcessingPool as Pool
from nltk import word_tokenize

def mrr_precision_at_k(golden, preds, k_list=[1,]):
    """
    Calculates Mean Reciprocal Error and Hits@1 == Precision@1
    """
    my_score = 0
    precision_at = np.zeros(len(k_list))
    for key, elem in enumerate(golden):
        if elem in preds[key]:
            location = np.where(preds[key]==elem)[0][0]
            my_score += 1/(1+ location)
        for k_index, k_value in enumerate(k_list):
            if location < k_value:
                precision_at[k_index] += 1
    return my_score/len(golden), precision_at/len(golden)


def clean_corpus_using_embeddings_vocabulary(word2keep, embeddings_dico, corpus, vectors, language, stops, instances = 10000):
    """
    Cleans corpus using the dictionary of embeddings. Any word without an associated embedding in the dictionary is ignored.
    Also, adds '__en' and '__fr' at the end of the words according to the language that they are written. That way, words that
    are the same across langueges (transition/transition in English and French) become different and the embeddings do not collapse.
    We investigate the effect of collaplsing such embeddings in the paper.
    """
    clean_corpus, clean_vectors, keys = [], {}, []
    words_we_want = set(embeddings_dico).difference(stops)
    for key, doc in enumerate(corpus[:1500]):
        clean_doc = []
        words = word_tokenize(doc)#.split()
        for word in words:
            word = word.lower()
            if word in words_we_want: 
                clean_doc.append(word+"__%s"%language)
                clean_vectors[word+"__%s"%language] = np.array(vectors[word].split()).astype(np.float)
            if len(clean_doc) == word2keep:
                break
        if len(clean_doc) > 5 :
            keys.append(key)
        clean_corpus.append(" ".join(clean_doc))
    return np.array(clean_corpus), clean_vectors, keys



def load_embeddings(path, dimension):
    """
    Loads the embeddings from a file with word2vec format. 
    The word2vec format is one line per words and its associated embedding.
    """
    f = codecs.open(path, encoding="utf8").read().splitlines()
    vectors = {}
    for i in f:
        elems = i.split()
        vectors[" ".join(elems[:-dimension])] =  " ".join(elems[-dimension:])
    return vectors


class WassersteinDistances(KNeighborsClassifier):
    """
    Implements a nearest neighbors classifier for input distributions using the Wasserstein distance as metric.
    Source and target distributions are l_1 normalized before computing the Wasserstein distance. 
    Wasserstein is parametrized by the distances between the individual points of the distributions.  
    In this work, we propose to use cross-lingual embeddings for calculating these distances.
        
    """
    def __init__(self, W_embed, n_neighbors=1, n_jobs=1, verbose=False, sinkhorn= False, sinkhorn_reg=0.1):
        """
        Initialization of the class.
        Arguments
        ---------
        W_embed: embeddings of the words, np.array
        verbose: True/False
        """
        self.sinkhorn = sinkhorn
        self.sinkhorn_reg = sinkhorn_reg
        self.W_embed = W_embed
        self.verbose = verbose
        super(WassersteinDistances, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs, metric='precomputed', algorithm='brute')

    def _wmd(self, i, row, X_train):
        union_idx = np.union1d(X_train[i].indices, row.indices)
        W_minimal = self.W_embed[union_idx]
        W_dist = euclidean_distances(W_minimal)
        bow_i = X_train[i, union_idx].A.ravel()
        bow_j = row[:, union_idx].A.ravel()
        if self.sinkhorn:
            return  ot.sinkhorn2(bow_i, bow_j, W_dist, self.sinkhorn_reg, numItermax=50, method='sinkhorn_stabilized',)[0]
        else:
            return  ot.emd2(bow_i, bow_j, W_dist)


    def _wmd_row(self, row):
        X_train = self._fit_X
        n_samples_train = X_train.shape[0]
        return [self._wmd(i, row, X_train) for i in range(n_samples_train)]

    def _pairwise_wmd(self, X_test, X_train=None):
        n_samples_test = X_test.shape[0]

        if X_train is None:
            X_train = self._fit_X
        pool = Pool(nodes=self.n_jobs) # Parallelization of the calculation of the distances
        dist  = pool.map(self._wmd_row, X_test)
        return np.array(dist)

    def fit(self, X, y):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        return super(WassersteinDistances, self).fit(X, y)

    def predict(self, X):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        return super(WassersteinDistances, self).predict(dist)

    def kneighbors(self, X, n_neighbors=1):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        return super(WassersteinDistances, self).kneighbors(dist, n_neighbors)

