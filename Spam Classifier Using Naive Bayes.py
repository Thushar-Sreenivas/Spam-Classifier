# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:58:22 2019

@author: Thushar Sreenivas
"""

import nltk
import pandas as pd


messages =  pd.read_csv(r'C:\Users\ASUS\.spyder-py3\SMSSpamCollection.txt',sep ='\t',
                         names = ['label','message'])

              
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
lem = []
ps = PorterStemmer()
wordnet=WordNetLemmatizer()


corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    rev = review
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #rev = [wordnet.lemmatize(word) for word in rev if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    rev = ' '.join(rev)
    corpus.append(review)
    #lem.append(rev)

# Bag of Words Model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()    
y = pd.get_dummies(messages['label'])
y =y.iloc[:,1].values

# TF-IDF Model
"""
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
z = tf.fit_transform(corpus).toarray()
"""
# train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 0)

# Training a model using Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)





