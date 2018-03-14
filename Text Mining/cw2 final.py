# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:25:21 2018

@author: JaZz-
"""

import urllib
import codecs
import os
import numpy as np
import pandas as pd
import gensim
import operator
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# # Read data
# =============================================================================
book_data = []
file_count = 1

for dirname in os.listdir(os.getcwd() + "\gap-html"):
    book_text = ""
    for filename in os.listdir(os.getcwd() + "\gap-html\\" + dirname):
        with open("./gap-html/" + dirname + "/" + filename, encoding="utf-8") as f:
            data = f.read()
            soup = BeautifulSoup(data, 'html.parser')
        
            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out
            
            # get text
            text = soup.get_text()
            
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # remove "OCR Output text"
            text = text.replace("OCR Output","")
            book_text = book_text +  "\n" + text
            
    book_data.append({"dirname": dirname, "text": book_text})
    file_count += 1
    
book_df = pd.DataFrame(book_data)

# =============================================================================
# Prepare data
# eg. tokenization
# =============================================================================
baseRegx="\w+|\!|\?"
tokenizer = RegexpTokenizer(baseRegx)

#Get english stopwords
eng_stopwords = stopwords.words('english') 
negative_words = ["aren","aren't","couldn","couldn't","didn","didn't","doesn","doesn't","don","don't","hadn","hadn't","hasn","hasn't","haven","haven't","isn","isn't","mightn","mightn't","mustn","mustn't","needn","needn't","no","nor","not","shan","shan't","should've","shouldn","shouldn't","wasn","wasn't","weren","weren't","won","won't","wouldn","wouldn't"]
stop_words_exclude_neg = list(set(eng_stopwords).difference(negative_words))

#Define Lemmatizer
lemmatizer = WordNetLemmatizer()

#Start pre-processing
tokenized_text = []
for text in book_df.text:
    #Lowercase
    lower_case = text.lower()
    
    #Tokenize
    tokens = tokenizer.tokenize(lower_case)
    
    #Re-initial token list in each round
    filtered_tokens=[] 
    
    #Remove stop word but include the negative helping verb
    for word in tokens:
        if not word in stop_words_exclude_neg:
            #Lemmatize 
            lemmatized = lemmatizer.lemmatize(word, pos="v")
            filtered_tokens.append(lemmatized)
        
    #Append each tokenized tweet in the list
    tokenized_text.append(filtered_tokens)

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.TaggedDocument(doc, [self.labels_list[idx]])

iter_tag_doc = LabeledLineSentence(tokenized_text, book_df.dirname)

# =============================================================================
# train data by doc2vec
# =============================================================================
# Create a Doc2Vec model
model = gensim.models.Doc2Vec(size=100, min_count=0
                              , alpha=0.025, min_alpha=0.025
                              , seed=0, workers=4)

# Build a set of vocabulary
model.build_vocab(iter_tag_doc)
print( 'number of vocabulary : ' + str(len(model.wv.vocab)) )

# Train the doc2vec model
for epoch in range(10):    # number of epoch
    #print( 'iteration '+str(epoch+1) )
    model.train(iter_tag_doc, total_examples=len(tokenized_text), epochs=1 )
    # Change learning rate for next epoch
    model.alpha -= 0.002
    model.min_alpha = model.alpha
print('model trained')

# =============================================================================
# # Test the model
# #to get most similar document with similarity scores using document- name
# sims = model.docvecs.most_similar("gap_-C0BAAAAQAAJ", topn=3)
# print('top similar document for gap_-C0BAAAAQAAJ: ')
# print(sims)
# 
# similar_words = model.docvecs.most_similar(positive=[model.docvecs['gap_2X5KAAAAYAAJ']])
# print(similar_words)
# =============================================================================

# =============================================================================
# find most frequent words
# =============================================================================
# word_count = 0
# for word, vocab_obj in sorted(model.wv.vocab.items(), key=operator.itemgetter(1), reverse=True):
#     print(str(word) + ": " + str(vocab_obj.count))
#     word_count += 1
#     if(word_count >= 100):
#         break
# =============================================================================


# =============================================================================
# Find distance between vectors
# =============================================================================
# predefinded matrix
dist = [[1 for x in range(24)] for y in range(24)]

# put similarity between each documents to matrix
for i in range(0, len(model.docvecs)):
    docvec = model.docvecs[i]
    sims = model.docvecs.most_similar(book_df.dirname[i], topn=23)
    for j in range(0, len(book_df)):
        for sim in sims:
            if(book_df.dirname[j] == sim[0]):
                dist[i][j] = sim[1]

# fixed the symmetric problems
dist = np.matrix(dist)
for i in range(0, 24):
    for j in range(0, 24):
        dist[j, i] = dist[i, j]

# find distance between vectors
dist = 1 - dist

# =============================================================================
# k means determine k
# =============================================================================
distortions = []
K = range(1, 23)
for k in K:
    kmeanModel = KMeans(n_clusters = k).fit(dist)
    distortions.append(sum(np.min(cdist(dist, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / dist.shape[0])
    
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# =============================================================================
# clustering data
# =============================================================================
num_clusters = 4

km = KMeans(n_clusters = num_clusters)

km.fit(dist)

clusters = km.labels_.tolist()

# =============================================================================
# MDS 100 to 2 dimensions
# =============================================================================
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

# set cluster color
color = []
for kmean in clusters:
    if(kmean == 0):
        color.append("red")
    elif(kmean == 1):
        color.append("green")
    elif(kmean == 2):
        color.append("blue")
    elif(kmean == 3):
        color.append("pink")
    elif(kmean == 4):
        color.append("magenta")
    else:
        color.append("yellow")


xs, ys = pos[:, 0], pos[:, 1]
plt.scatter(xs, ys, c = color, s=xs.shape[0], cmap='viridis')
plt.title('Clusters')
for i in range(0, xs.shape[0]):
    plt.text(xs[i], ys[i], str(i + 1))
plt.show()