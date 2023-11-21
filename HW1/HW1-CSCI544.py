#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
#nltk.download('wordnet')
import re

#!pip install bs4
#!pip install contractions 
#!pip install scikit-learn

import contractions
from bs4 import BeautifulSoup

#remove warnings in output
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[3]:


#skipping rows in dataset which give error 
#reference: https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data
df = pd.read_table("amazon_reviews_us_Beauty_v1_00.tsv", on_bad_lines='skip')


# ## Keep Reviews and Ratings

# In[4]:


df = df[["star_rating", "review_body"]]
#df.head()


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[5]:


#adding new column named "class" to define class 1,2 and 3
#reference: https://sparkbyexamples.com/pandas/pandas-apply-with-lambda-examples/#:~:text=Apply%20Lambda%20Expression%20to%20Single,x%3Ax%2D2)%20
df["class"] = df["star_rating"].apply(lambda x : 3 if str(x) > '3' else 2 if str(x) == '3' else 1)

#select 20000 reviews from each class
#reference: https://stackoverflow.com/questions/67174746/sklearn-take-only-few-records-from-each-target-class
df = df.groupby('class').sample(n=20000, replace=True)


# # Data Cleaning
# 
# 

# # Pre-processing

# In[6]:


df_train = df[["review_body", "class"]]

charlenpre = df_train['review_body'].str.len().mean()

#review to lower case
df_train['review_body'] = df_train['review_body'].apply(lambda x : str(x).lower())

#remove html tags
#reference: https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
df_train['review_body'] = df_train['review_body'].apply(lambda x: BeautifulSoup(str(x)).get_text())

#remove url
#reference: https://stackoverflow.com/questions/51994254/removing-url-from-a-column-in-pandas-dataframe
df_train['review_body'] = df_train['review_body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

#remove non-alphabetical words
df_train['review_body'] = df_train['review_body'].replace('[^a-zA-Z ]', '', regex=True)

#remove extra spaces
df_train['review_body'] = df_train['review_body'].str.strip()

#perform contractions
#reference: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
df_train['review_body'] = df_train['review_body'].apply(lambda x: contractions.fix(str(x)))

charlenpost = df_train['review_body'].str.len().mean()

print("Average review character length before and after cleaning: ", charlenpre, ",", charlenpost)




# ## remove the stop words 

# In[7]:


from nltk.corpus import stopwords 

#nltk.download('stopwords')

charlenpre = df_train['review_body'].str.len().mean()

stopwords = stopwords.words('english')


#remove the words which are present in stopwords
#reference: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
df_train['review_body'] = df_train['review_body'].apply(lambda x: ' '.join([word for word in x.split() 
                                                                            if word not in (stopwords)]))


# ## perform lemmatization  

# In[8]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#perform lemmatization of words
#reference: 1. https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
#           2. https://www.nltk.org/_modules/nltk/stem/wordnet.html
lemmatizer = WordNetLemmatizer()

#verbs
df_train['review_body'] = df_train['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos="v") 
                                                                            for word in x.split()]))

#noun
df_train['review_body'] = df_train['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos="n") 
                                                                            for word in x.split()]))

#adjectives
df_train['review_body'] = df_train['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos="a") 
                                                                            for word in x.split()]))

#adverbs
df_train['review_body'] = df_train['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos="r") 
                                                                            for word in x.split()]))

#satellite adjectives
df_train['review_body'] = df_train['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos="s") 
                                                                            for word in x.split()]))

charlenpost = df_train['review_body'].str.len().mean()

print("Average review character length before and after pre-processing: ",charlenpre, ", ", charlenpost)

#df_train.head(50)


# # TF-IDF Feature Extraction

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
 
#tf-idf feature extraction of input
#reference: https://stackoverflow.com/questions/37593293/how-to-get-tfidf-with-pandas-dataframe
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(df_train['review_body'])

#vector


# # Perceptron

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

#split data into training and test
#reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
Xtrain, Xtest, Ytrain, Ytest = train_test_split(vector, df_train['class'], stratify=df_train['class'], 
                                                test_size=0.2, random_state=42)

#Perceptron model training
#reference: https://python-course.eu/machine-learning/perceptron-class-in-sklearn.php
model_p = Perceptron(random_state=42)
model_p.fit(Xtrain, Ytrain)

#Testing the model
Ypred = model_p.predict(Xtest)

precision_score_p = precision_score(Ytest, Ypred, average=None)
recall_score_p = recall_score(Ytest, Ypred, average=None)
f1_score_p = f1_score(Ytest, Ypred, average=None)

print("Perceptron model output:")
print("class1: ", precision_score_p[0], ", ", recall_score_p[0], ", ", f1_score_p[0])
print("class2: ", precision_score_p[1], ", ", recall_score_p[1], ", ", f1_score_p[1])
print("class3: ", precision_score_p[2], ", ", recall_score_p[2], ", ", f1_score_p[2])
print("average:", precision_score(Ytest, Ypred, average='weighted'),", ", recall_score(Ytest, Ypred, average='weighted'),
     ", ", f1_score(Ytest, Ypred, average='weighted'))


#print("training dataresults: ")
#print(classification_report(model_p.predict(Xtrain), Ytrain))

#print("testing dataresults: ")
#print(classification_report(Ypred, Ytest))


# # SVM

# In[11]:


from sklearn.svm import LinearSVC

#Linear SVM model training
#reference: 1. https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
#           2. https://stackoverflow.com/questions/27912872/what-is-the-difference-between-svc-and-svm-in-scikit-learn
model_s = LinearSVC(random_state=42) 
model_s.fit(Xtrain, Ytrain)

#Testing the model
Ypred = model_s.predict(Xtest)

precision_score_s = precision_score(Ytest, Ypred, average=None)
recall_score_s = recall_score(Ytest, Ypred, average=None)
f1_score_s = f1_score(Ytest, Ypred, average=None)

print("SVM model output:")
print("class1: ", precision_score_s[0], ", ", recall_score_s[0], ", ", f1_score_s[0])
print("class2: ", precision_score_s[1], ", ", recall_score_s[1], ", ", f1_score_s[1])
print("class3: ", precision_score_s[2], ", ", recall_score_s[2], ", ", f1_score_s[2])
print("average:", precision_score(Ytest, Ypred, average='weighted'), ", ", recall_score(Ytest, Ypred, average='weighted'), 
      ", ", f1_score(Ytest, Ypred, average='weighted'))


#print("training dataresults: ")
#print(classification_report(model_s.predict(Xtrain), Ytrain))

#print("testing dataresults: ")
#print(classification_report(Ypred, Ytest))


# # Logistic Regression

# In[12]:


from sklearn.linear_model import LogisticRegression

#Logistic Regression model training
#reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
model_l = LogisticRegression(random_state=42) 
model_l.fit(Xtrain, Ytrain)

#Testing the model
Ypred = model_l.predict(Xtest)

precision_score_l = precision_score(Ytest, Ypred, average=None)
recall_score_l = recall_score(Ytest, Ypred, average=None)
f1_score_l = f1_score(Ytest, Ypred, average=None)

print("Logistic Regression model output:")
print("class1: ", precision_score_l[0], ", ", recall_score_l[0], ", ", f1_score_l[0])
print("class2: ", precision_score_l[1], ", ", recall_score_l[1], ", ", f1_score_l[1])
print("class3: ", precision_score_l[2], ", ", recall_score_l[2], ", ", f1_score_l[2])
print("average:", precision_score(Ytest, Ypred, average='weighted'),", ", recall_score(Ytest, Ypred, average='weighted'), 
      ", ", f1_score(Ytest, Ypred, average='weighted'))


#print("training dataresults: ")
#print(classification_report(model_l.predict(Xtrain), Ytrain))

#print("testing dataresults: ")
#print(classification_report(Ypred, Ytest))


# # Naive Bayes

# In[13]:


from sklearn.naive_bayes import MultinomialNB

model_n = MultinomialNB() 
model_n.fit(Xtrain, Ytrain)

#Testing the model
Ypred = model_n.predict(Xtest)

precision_score_n = precision_score(Ytest, Ypred, average=None)
recall_score_n = recall_score(Ytest, Ypred, average=None)
f1_score_n = f1_score(Ytest, Ypred, average=None)

print("Multinomial Naive Bayes model output:")
print("class1: ", precision_score_n[0], ", ", recall_score_n[0], ", ", f1_score_n[0])
print("class2: ", precision_score_n[1], ", ", recall_score_n[1], ", ", f1_score_n[1])
print("class3: ", precision_score_n[2], ", ", recall_score_n[2], ", ", f1_score_n[2])
print("average:", precision_score(Ytest, Ypred, average='weighted'),", ", recall_score(Ytest, Ypred, average='weighted'), 
      ", ", f1_score(Ytest, Ypred, average='weighted'))

#print("training dataresults: ")
#print(classification_report(model_n.predict(Xtrain), Ytrain))

#print("testing dataresults: ")
#print(classification_report(Ypred, Ytest))

