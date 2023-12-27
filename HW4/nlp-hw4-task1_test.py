#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install torch

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#remove warnings in output
import warnings
warnings.filterwarnings('ignore')


# #### Read Data

# In[2]:


df_train = pd.read_table("data/train", delimiter=r"\s", header=None, names = ['index', 'word', 'NER_tag'])
df_dev = pd.read_table("data/dev", delimiter=r"\s", header=None, names = ['index', 'word', 'NER_tag'])
df_test = pd.read_table("data/test", delimiter=r"\s", header=None, names = ['index', 'word'])


# # Task 1: Simple Bidirectional LSTM model

# <b>word and tags vocab</b>

# In[3]:


word_vocab = (df_train['word'].unique()).tolist()
tag_vocab = (df_train['NER_tag'].unique()).tolist()

word_vocab.append('<UNK>')

word_to_index = {w:i for i,w in enumerate(word_vocab, 1)}
tag_to_index = {t:i for i,t in enumerate(tag_vocab)}

index_to_tag = {v: k for k, v in tag_to_index.items()}


# In[4]:


tag_to_index


# In[5]:


class BLSTM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, lstm_layers, lstm_dropout, linear_output_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim, padding_idx = 0)
        self.blstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, batch_first = True, num_layers=lstm_layers, bidirectional=True)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_output_dim)
        self.dropout = nn.Dropout(p = lstm_dropout)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, num_classes)

    def forward(self, X, lengths):
        
        embedded = self.embedding(X)       
#         packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)    
        blstm_output, _ = self.blstm(embedded)
#         output, _ = nn.utils.rnn.pad_packed_sequence(blstm_output, batch_first=True, padding_value=-1)     
        output = self.dropout(blstm_output)     
        linear_output = self.activation(self.linear(output))      
        logits = self.classifier(linear_output)
        
        return logits


# In[7]:


vocab_size = len(word_to_index) + 1
num_classes = len(tag_to_index)

model_loaded = BLSTM(vocab_size=vocab_size, embedding_dim=100, lstm_hidden_dim=256, lstm_layers=1, lstm_dropout=0.33, linear_output_dim=128, num_classes = num_classes)
state_dict = torch.load('blstm1.pt', map_location=torch.device('cpu'))
model_loaded.load_state_dict(state_dict)
model_loaded.eval()


# In[14]:


sentence_list = []
sentence = []

for i in df_dev.index:

    word = df_dev['word'][i]

    if df_dev['index'][i] == 1:
        if(i != 0):
            sentence_list.append(sentence)
        sentence = []

    sentence.append(word)

if sentence:
    sentence_list.append(sentence)
    


# In[16]:


model_loaded.eval()

f = open("dev1.out", "w")

for sentence in sentence_list:

    s= [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence]
    input_tensor = torch.tensor([s])
    lengths = [len(s)]
 
    with torch.no_grad():
        output = model_loaded(input_tensor, lengths)
        output = output.view(-1, 9)
        
    _, predicted = torch.max(output, 1)

    predicted_labels_dev = [index_to_tag[i.item()] for i in predicted]

    i = 0
    for token, label in zip(sentence, predicted_labels_dev):
        i += 1
        text = str(i) + " " + str(token) + " " + str(label) + "\n"
        f.write(text)

    f.write("\n")

f.close()


# In[11]:


sentence_list = []
sentence = []

unknown_index = word_to_index['<UNK>']

for i in df_test.index:

    word = df_test['word'][i]

    if df_test['index'][i] == 1:
        if(i != 0):
            sentence_list.append(sentence)
        sentence = []

    sentence.append(word)

if sentence:
    sentence_list.append(sentence)
    


# In[13]:


model_loaded.eval()

f = open("test1.out", "w")

for sentence in sentence_list:

    s= [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence]
    input_tensor = torch.tensor([s])
    lengths = [len(s)]
 
    with torch.no_grad():
        output = model_loaded(input_tensor, lengths)
        output = output.view(-1, 9)
        
    _, predicted = torch.max(output, 1)

    predicted_labels_test = [index_to_tag[i.item()] for i in predicted]

    i = 0
    for token, label in zip(sentence, predicted_labels_test):
        i += 1
        text = str(i) + " " + str(token) + " " + str(label) + "\n"
        f.write(text)

    f.write("\n")

f.close()


# In[ ]:




