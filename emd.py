from __future__ import division
from scipy import sparse
from wass_funcs import  load_embeddings, mrr_precision_at_k, clean_corpus_using_embeddings_vocabulary, WassersteinDistances 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np, sys, codecs, nltk.data
from sklearn.preprocessing import normalize

vectors_en = load_embeddings(sys.argv[1], 300) # Load vectors for lang. 1, English by default
vectors_fr = load_embeddings(sys.argv[2], 300) # Load vectors for lang. 2

en  = codecs.open(sys.argv[3], encoding="utf8").read().splitlines() # Open docs of lang. 1 as utf8, read them as a list of documents
fr = codecs.open(sys.argv[4], encoding="utf8").read().splitlines() # Same for lang. 2.. Encoding here is important

word2keep = int(sys.argv[5]) # Max words of each document. Will be used later for speeding calculations.
if sys.argv[6]=="greek": # There is an issue with \sigma in the embeddings. Greeks have different sigma symbol in the word or in the end 
    fr = [i.replace(u"\u03c2", u"\u03c3") for i in fr]

clean_en, clean_vectors_en, keys_en = clean_corpus_using_embeddings_vocabulary(word2keep, set(vectors_en.keys()), en, vectors_en, "en", set(nltk.corpus.stopwords.words("english")))
clean_fr, clean_vectors_fr, keys_fr = clean_corpus_using_embeddings_vocabulary(word2keep, set(vectors_fr.keys()), fr, vectors_fr, "fr", set(nltk.corpus.stopwords.words(sys.argv[6])))

common_keys = set(keys_en).intersection(set(keys_fr)) # Article ids that have more than 5 words in each language..
common_keys = np.array(list(common_keys)) # ids of long docs for both languages
common_keys = common_keys[:500] # Make the CLDR experiement with 500 documents
instances = len(common_keys)
clean_en, clean_fr = list(clean_en[common_keys]), list(clean_fr[common_keys]) # clean_en/fr are the two corpora that will be used for CLDR

print "Starting with documents of size", len(clean_en), len(clean_fr)

del vectors_fr, vectors_en, en, fr # to save space in memory

vec1 = CountVectorizer().fit(clean_en+clean_fr) # get the vocabulary of the corpus

common = [word for word in vec1.get_feature_names() if word in clean_vectors_en or word in clean_vectors_fr] # Keep words with associated embeddings e.g., get the fixed vocabulary
W_common= []
for w in common: # Similarly, to save memory keep only the embeddings of words that appear in the corpus. 
    if w in clean_vectors_en:
        W_common.append(np.array(clean_vectors_en[w]) )
    else:
        W_common.append(np.array(clean_vectors_fr[w]) )
del clean_vectors_en, clean_vectors_fr # Remove the rest of the embeddings
print "The vocabulary size is:", len(W_common)
W_common = np.array(W_common) 
W_common = normalize(W_common)
vect = TfidfVectorizer(vocabulary=common, dtype=np.double, norm=None, )# Tf-idf representation of docs in both lang. 1 and lang. 2
vect.fit(clean_en+clean_fr)
X_train_idf = vect.transform(clean_en) # Lang. 1: tf-idf representation of the corpus
X_test_idf = vect.transform(clean_fr) # Lang. 2: tf-idf representation of the corpus 

vect_tf = CountVectorizer(vocabulary=common, dtype=np.double)
vect_tf.fit(clean_en+clean_fr)
X_train_tf = vect_tf.transform(clean_en) # Lang. 1: tf representation of the corpus
X_test_tf = vect_tf.transform(clean_fr) # Lang. 2: tf representation of the corpus

######################################################################################
print "Starting experiments with WMD - tfidf: Retrieve English documents, given queries in %s"% sys.argv[6]
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10)
clf.fit(X_train_idf[:instances], np.ones(instances))
dist, preds = clf.kneighbors(X_test_idf[:instances], n_neighbors=instances)
print("Scores:", mrr_precision_at_k(range(len(preds)), preds))

print "Starting experiments with Sinkhorn - tfidf: Retrieve English documents, given queries in %s"% sys.argv[6] 
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10, sinkhorn=True)
clf.fit(X_train_idf[:instances], np.ones(instances))
dist, preds = clf.kneighbors(X_test_idf[:instances], n_neighbors=instances)
print("Scores:", mrr_precision_at_k(range(len(preds)), preds))

print "Starting experiments with WMD - tf: Retrieve English documents, given queries in %s"% sys.argv[6]
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10)
clf.fit(X_train_tf[:instances], np.ones(instances))
dist, preds = clf.kneighbors(X_test_tf[:instances], n_neighbors=instances)
print("Scores:", mrr_precision_at_k(range(len(preds)), preds))

print "Starting experiments with Sinkhorn - tf: Retrieve English documents, given queries in %s"% sys.argv[6]
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10, sinkhorn=True)
clf.fit(X_train_tf[:instances], np.ones(instances))
dist, preds = clf.kneighbors(X_test_tf[:instances], n_neighbors=instances)
print("Scores:", mrr_precision_at_k(range(len(preds)), preds))
#####################################################################################3
print  "Starting experiments with WMD - tfidf: Retrieve %s documents, given queries in English."% sys.argv[6]
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10)
clf.fit(X_test_idf[:instances], np.ones(instances))
dist, preds = clf.kneighbors(X_train_idf[:instances], n_neighbors=instances)
print("Scores:", mrr_precision_at_k(range(len(preds)), preds))

print "Starting experiments with Sinkhorn - tfidf: Retrieve %s documents, given queries in English."% sys.argv[6]
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10, sinkhorn=True)
clf.fit(X_test_idf[:instances], np.ones(instances))
dist, preds = clf.kneighbors(X_train_idf[:instances], n_neighbors=instances)
print("Scores:", mrr_precision_at_k(range(len(preds)), preds))

print "Starting experiments with WMD - tf: Retrieve %s documents, given queries in English."% sys.argv[6]
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10)
clf.fit(X_test_tf[:instances], np.ones(instances))
dist, preds = clf.kneighbors(X_train_tf[:instances], n_neighbors=instances)
print("Scores:", mrr_precision_at_k(range(len(preds)), preds))

print "Starting experiments with Sinkhorn - tf: Retrieve %s documents, given queries in English."% sys.argv[6]
clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10, sinkhorn=True)
clf.fit(X_test_tf[:instances], np.ones(instances))
dist, preds = clf.kneighbors(X_train_tf[:instances], n_neighbors=instances)
print("Scores:", mrr_precision_at_k(range(len(preds)), preds))

