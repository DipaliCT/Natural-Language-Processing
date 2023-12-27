#!/usr/bin/env python
# coding: utf-8

# In[25]:


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

# In[26]:


df_train = pd.read_table("data/train", delimiter=r"\s", header=None, names = ['index', 'word', 'NER_tag'])
df_dev = pd.read_table("data/dev", delimiter=r"\s", header=None, names = ['index', 'word', 'NER_tag'])
df_test = pd.read_table("data/test", delimiter=r"\s", header=None, names = ['index', 'word'])


# # Task 1: Simple Bidirectional LSTM model

# <b>word and tags vocab</b>

# In[28]:


word_vocab = (df_train['word'].unique()).tolist()
tag_vocab = (df_train['NER_tag'].unique()).tolist()

word_vocab.append('<UNK>')

word_to_index = {w:i for i,w in enumerate(word_vocab, 1)}
tag_to_index = {t:i for i,t in enumerate(tag_vocab)}

index_to_tag = {v: k for k, v in tag_to_index.items()}


# In[29]:


tag_to_index


# In[30]:


def get_data(df):
    sentence_list = []
    tag_list = []
    sentence = []
    tags = []
    X_train = []
    Y_train = []
    
    unknown_index = word_to_index['<UNK>']

    for i in df.index:

        word = df['word'][i]
        tag = df['NER_tag'][i]

        if df['index'][i] == 1:
            if(i != 0):
                sentence_list.append(sentence)
                tag_list.append(tags)

            sentence = []
            tags = []

        sentence.append(word)
        tags.append(tag)

    if sentence:
        sentence_list.append(sentence)
        tag_list.append(tags)
        
    X_index = [[word_to_index.get(word, unknown_index) for word in sentence] for sentence in sentence_list]
    Y_index = [[tag_to_index[tag] for tag in tags] for tags in tag_list]
    lengths = [len(sentence) for sentence in sentence_list]
    
    for X_list in X_index:
        tensor = torch.tensor(X_list)
        X_train.append(tensor)

    for Y_list in Y_index:
        tensor = torch.tensor(Y_list)
        Y_train.append(tensor)
    
    return X_train, Y_train, lengths


# In[31]:


X_train, Y_train, lengths_train = get_data(df_train)
X_dev, Y_dev, lengths_dev = get_data(df_dev)


# In[32]:


class NERDataset(Dataset):
    def __init__(self, Dataset):
        self.X = nn.utils.rnn.pad_sequence(Dataset[0], batch_first = True, padding_value = 0)
        self.Y = nn.utils.rnn.pad_sequence(Dataset[1], batch_first = True, padding_value = -1)
        self.lengths = Dataset[2]

    def __getitem__(self, idx):
        item = (self.X[idx], self.Y[idx], self.lengths[idx])
        return item

    def __len__(self):
        return len(self.X)


# In[33]:


batch_size = 8

train_data = NERDataset([X_train, Y_train, lengths_train])
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

dev_data = NERDataset([X_dev, Y_dev, lengths_dev])
dev_data_loader = DataLoader(dev_data, batch_size = batch_size, shuffle = False)


# In[34]:


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


# In[35]:


weights = torch.tensor([1,0.6,1,1,1,1,1,1,1])

vocab_size = len(word_to_index) + 1
num_classes = len(tag_to_index)
total_epochs = 30

model = BLSTM(vocab_size=vocab_size, embedding_dim=100, lstm_hidden_dim=256, lstm_layers=1, lstm_dropout=0.33, linear_output_dim=128, num_classes = num_classes)
    
criterion = nn.CrossEntropyLoss(weight = weights, ignore_index = -1).cuda()
optimizer = optim.SGD(model.parameters(), lr=1)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1,
#                                                 total_steps=total_epochs*(len(X_train)//batch_size + 1),
#                                                 anneal_strategy='linear')


# In[36]:


for epoch in range(total_epochs):
    
    model.cuda()
    model.train()

    train_loss = 0
    
    predicted_labels = []

    for idx, batch_data in enumerate(train_data_loader):

        optimizer.zero_grad()
        
        X_train = (batch_data[0]).cuda()
        Y_train = (batch_data[1]).cuda()
        lengths_train = batch_data[2]
        
        Y_pred = model(X_train, lengths_train)
        
        Y_pred = Y_pred.view(-1, 9)
        Y_train = Y_train.view(-1)
       
        loss = criterion(Y_pred, Y_train)       
        loss.backward()
        
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= len(train_data_loader)
    
    model.cuda()
    model.eval()
    
    val_loss = 0
    
    with torch.no_grad():
        for idx, batch_data in enumerate(dev_data_loader):

            X_dev = (batch_data[0]).cuda()
            Y_dev = (batch_data[1]).cuda()
            lengths_dev = batch_data[2]

            Y_pred_dev = model(X_dev, lengths_dev)

            Y_pred_dev = Y_pred_dev.view(-1, 9)
            Y_dev = Y_dev.view(-1)

            loss = criterion(Y_pred_dev, Y_dev)  

            val_loss += loss.item()
            
        val_loss /= len(dev_data_loader)
        
    scheduler.step(val_loss)
    
    print('Epoch [%d], training loss: %.4f, validation loss: %.4f' % (epoch + 1, train_loss, val_loss))


# In[40]:


torch.save(model.state_dict(), 'blstm1.pt')

