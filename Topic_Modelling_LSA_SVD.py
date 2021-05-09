# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:41:47 2020

@author: akadali
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
stops = nltk.corpus.stopwords.words('english')

tf_idf_vectorizer = TfidfVectorizer(max_df = 0.99, 
                                    max_features = 3000, 
                                    stop_words = stops, 
                                    use_idf = True, 
                                    ngram_range = (1,3))
# Read in the cleaned data, before the CountVectorizer step
data_clean = pd.read_pickle('data_clean.pkl')

tf_idf_matrix = tf_idf_vectorizer.fit_transform(data_clean)

tf_idf_matrix.shape

from sklearn.decomposition import TruncatedSVD

"""
The next step is to represent each and every term and document as a vector. We will use the document-term 
matrix and decompose it into multiple matrices. We will use sklearn’s TruncatedSVD to perform the task of matrix 
decomposition.
"""
# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components = 2, n_iter= 7, random_state = 122)
svd_model.fit(tf_idf_matrix)

svd_model.components_

len(svd_model.components_)

"""
The components of svd_model are our topics, and we can access them using svd_model.components_. 
Finally, let’s print a few most important words in each of the 2 topics and see how our model has done.
"""
terms = tf_idf_vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic#"+str(i)+": ")
    print(" ")
    for t in sorted_terms:
        print(t[0])
    print(" ")

"""
Topics Visualization
To find out how distinct our topics are, we should visualize them. 
Of course, we cannot visualize more than 3 dimensions, but there are techniques like PCA and t-SNE which can 
help us visualize high dimensional data into lower dimensions. Here we will use a relatively new technique 
called UMAP (Uniform Manifold Approximation and Projection).
"""

import pyLDA


import umap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

X_topics = svd_model.fit_transform(tf_idf_matrix)
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], 
            embedding[:, 1],
            s = 10, # size
            edgecolor='none')




