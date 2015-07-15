
# coding: utf-8

# In[11]:

#############
#Import List#
#############

#Regular expressions to help parse the document text#

import re

# lda (Latent Dirchlet Allocation) to do topic modeling
from lda import LDA
# TermDocumentMatrix, to create a term document matrix. 
# This library also tokenizes the corpus and removes punctuation
from textmining import TermDocumentMatrix
# Making sure te tokenize function removes stopwords. 
# Later I would like to use TFIDF
from textmining import simple_tokenize_remove_stopwords
#lda uses numpy arrays as input 
import numpy as np
#copy because python does not know how to copy things
import copy
#json will be our output format. 
import json

############################################################################
# RunLDA Runs Limited Dirschlet Topic Modeling on the Corpus of Articles   
# Released by the Washington Post During Hacking Journalism DC 2015        
# As of Now, it just prints out the topics                                 
#                                                                          
# Parameters:
# 
#      FileLocation <string> - The File location of the data set. 
#                              This should be set to
#                              '\Hackathon2015_WashingtonPost\content.txt'
#                              In Whatever directory your WasPost data
#                              is in
#
#      NumDocs - The number of Articles to run over, 
#                 Starting from the beginning
#
#      NumTopics - The number of topics to look create
############################################################################
def RunLDA(FileLocation, NumDocs, NumTopics):
    # In order to create a Term Document matrix,
    # We read in every file and then make a list containing the body of 
    # all of the articles
    fin=open(FileLocation,'r')
    #Will need to store the urls when we make the tdm
    UrlArray = []
    #Create TDM object. It will also remove stopwords
    TDM = TermDocumentMatrix(simple_tokenize_remove_stopwords)
    # Add each article to the TDM object. Also create a list of urls
    # This is a massive corpus, so we are only doing this for 300 articles.
    for i in range(NumDocs):
        Article = fin.next()
        UrlArray.append(re.split(r'\t',Article)[0])
        TDM.add_doc(re.split(r'\t',Article)[1])
    # Rows in TDM is an iterable 
    # We can't have that to input it into numpy
    X = list(TDM.rows())
    # Oddly enough the first row of the .rows() iterable in TDM returms a 
    # List of all of the words used. Think of it as a header file
    Vocab = X[0]
    Y = []
    #creating a 2d list containing the rows of the document matrix
    for i in range(len(X)-1):
        Y.append(X[i+1])
    # Create the LDA model object. 20 topics this time, but that can be changed. 
    model = LDA(n_topics=20, n_iter=1500, random_state=1)
    # Make a numpy Array to use as input
    Yarray = np.asarray(Y)
    #Fit the model. This process is similiar to scikit-learn's algorithms
    model.fit(Yarray)
    TopicWords = []
    topic_word = model.topic_word_
    n_top_words = 50
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(Vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        TopicWords.append(topic_words)
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))    


# In[12]:




# In[ ]:



