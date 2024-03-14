import os
import csv
import pandas as pd
import matplotlib.pyplot as plt        
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection,naive_bayes,svm
from sklearn.metrics import accuracy_score
from scipy import stats

data = pd.read_csv('file.csv',encoding = 'latin-1')

import nltk
nltk.download("popular")

#Pre-processing steps

#Remove blank lines if any
data.dropna(inplace = True)

#Convert upper case to lower
data['Document'] = data['Document'].str.lower() 

#Tokenization
data['Document'] = [ word_tokenize(entry) for entry in data['Document'] ]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(data['Document']):
 final_words = []
 word_lem = WordNetLemmatizer()
 for word,tag in pos_tag(entry):
     if word not in stopwords.words('english') and word.isalpha():
         f_word = word_lem.lemmatize(word,tag_map[tag[0]])
         final_words.append(f_word)
 data.loc[index,'text_final'] = str(final_words)

X_train,X_test,y_train,y_test = train_test_split(data['text_final'],data['Topic'],test_size = 0.33,random_state = 42)

#TF-idf vectorizer
Tfidf_vect = TfidfVectorizer(max_features=None)
Tfidf_vect.fit(data['text_final'])
X_train_final = Tfidf_vect.transform(X_train).toarray()
X_test_final = Tfidf_vect.transform(X_test).toarray()


from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

#RMDL model
from RMDL import RMDL_Text as RMDL
RMDL.Text_Classification(X_train, y_train, X_test,  y_test, batch_size=128,
                 EMBEDDING_DIM=50,MAX_SEQUENCE_LENGTH = 500, MAX_NB_WORDS = 100000,
                 GloVe_dir="", GloVe_file = "glove.6B.50d.txt",
                 sparse_categorical=True, random_deep=[3, 3, 3], epochs=[20,10,10],  plot=True,
                 min_hidden_layer_dnn=1, max_hidden_layer_dnn=20, min_nodes_dnn=200, max_nodes_dnn=1024,
                 min_hidden_layer_rnn=3, max_hidden_layer_rnn=5, min_nodes_rnn=200,  max_nodes_rnn=800,
                 min_hidden_layer_cnn=3, max_hidden_layer_cnn=5, min_nodes_cnn=200, max_nodes_cnn=800,
                 random_state=42, random_optimizor=True, dropout=0.05)


