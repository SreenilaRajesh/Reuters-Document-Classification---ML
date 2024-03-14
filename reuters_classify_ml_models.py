import os
import csv
import pandas as pd        
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes,svm
from sklearn.metrics import accuracy_score


path = r"E:\packages\ML\Reuters21578-Apte-90Cat\training"

#list of various topics
entries = os.listdir(path)
  
with open("file.csv", 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile,dialect="excel") 
    # writing the fields 
    csvwriter.writerow(["Topic","Document"])     
    # writing the data rows
    for i in range(len(entries)):#len(entries)
        #entry contains various files under entries[i]
        entry = os.listdir(path + "\\" + entries[i]) 
        for j in range(len(entry)):
            #opening each file in entry,their contents are read and written to csv
            f=open(r"E:\packages\ML\Reuters21578-Apte-90Cat\training"+"\\" + entries[i]+"\\" + entry[j], "r")
            contents = f.read()
            #print(contents)
            app=[]
            app.append(entries[i])
            app.append(contents)
            #print(app)
            csvwriter.writerow(app)
            f.close()
csvfile.close()

data = pd.read_csv(r'file.csv',encoding = 'latin-1') 

#Pre-processing steps

#Remove blank lines if any
data.dropna(inplace = True)

#Convert upper case to lower
data['Document'] = data['Document'].str.lower() 

#Tokenization
data['Document'] = [ word_tokenize(entry) for entry in data['Document'] ]

#USing Lemmatization and converting all words to their base form to reduce word density

#A Dictionary is created with pos_tag as the keys
#whose values are mapped with the value from wordnet dictionary
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(data['Document']):
    final_words = []
    word_lem = WordNetLemmatizer()
    for word,tag in pos_tag(entry):
        #checking for non-stop words and checks for the word with only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            f_word = word_lem.lemmatize(word,tag_map[tag[0]])
            final_words.append(f_word)
    data.loc[index,'text_final'] = str(final_words)
    

 #Tf-idf vectorization is done   
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data['text_final'])
data['text_final']= Tfidf_vect.transform(data['text_final']).toarray()


from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np

#Multinomial Naive Bayes and SVM models are used to fit the given data
Naive = naive_bayes.MultinomialNB()
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
kf = KFold(n_splits=2)
kf.get_n_splits(data['text_final'])
TP_nbl=[]
TP_svml=[]
nb_accuracyl=[]
svm_accuracyl=[]
#The training data is split into Kfolds of train and test
for train_index, test_index in kf.split(data['text_final']):
    X_train_final, X_test_final = data['text_final'][train_index], data['text_final'][test_index]
    y_train, y_test = data['Topic'][train_index], data['Topic'][test_index]
    Naive.fit(X_train_final,y_train)
    pred = Naive.predict(X_test_final)
    #accuracy and true positive value of nb model from each fold is added to a list
    nb_accuracy=accuracy_score(pred,y_test)
    CM = confusion_matrix(y_test, pred)
    TP_nb = CM[1][1]
    TP_nbl.append(TP_nb)
    nb_accuracyl.append(nb_accuracy)

    SVM.fit(X_train_final,y_train)
    pred = SVM.predict(X_test_final)
    #accuracy and true positive value of svm from each fold is added to a list
    svm_accuracy=accuracy_score(pred,y_test)
    CM = confusion_matrix(y_test, pred)
    TP_svm = CM[1][1]
    TP_svml.append(TP_svm)
    svm_accuracyl.append(svm_accuracy)
# mean accuracy and tp from kfolds is found
TP_nb=np.mean(TP_nbl)
TP_svm=np.mean(TP_svml)
nb_accuracy=np.mean(nb_accuracyl)
svm_accuracy=np.mean(svm_accuracy)
print("Naive Bayes Accuracy Score:",nb_accuracy)
print("SVM Accuracy Score:",svm_accuracy)


#comparing the models using t-test using their true positive values
from scipy import stats
import numpy as np
p_cap1 = (TP_nb/len(X_test_final))
p_cap2 = (TP_svm/len(X_test_final))
 
p_cap = (TP_nb+TP_svm)/(2*len(X_test_final))
Z = (p_cap1-p_cap2)/np.sqrt(2*p_cap*(1-p_cap)/len(X_test_final))

p_value = stats.norm.pdf(abs(Z))*2
print(Z," ",p_value)



