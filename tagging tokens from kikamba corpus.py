#!/usr/bin/env python
# coding: utf-8

# In[49]:


training_file = open("corpus", "r")
all_lines = training_file.readlines()
training_file.close()


# In[50]:


#represent all lines as tokens
import nltk


# In[51]:


for e in all_lines:
    print (e.split(' '))
    


# In[52]:


def split_words():
    for words in all_lines:
        split_words=words.split(' ')
    return split_words


# In[53]:


tagged=list(split_words())


# In[54]:


tagged


# In[55]:


tagged_tokens=[nltk.tag.str2tuple(t) for t in tagged]


# In[56]:


tagged_tokens


# In[57]:


import re, pprint
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


# In[58]:


random.seed(1234)
train_set, test_set = train_test_split(tagged_tokens,test_size=0.3)
print("Train size:",len(train_set))
print("Test size",len(test_set))


# In[59]:


print(train_set[:5])


# In[60]:


# Getting list of tagged words
train_tagged_words = [tup for sent in train_set for tup in tagged_tokens]
len(train_tagged_words)


# In[61]:


# tokens 
tokens = [pair[0] for pair in train_tagged_words]
tokens[:10]


# In[62]:


# vocabulary
V = set(tokens)
print(len(V))


# In[63]:


# number of tags
T = set([pair[1] for pair in train_tagged_words])
len(T)


# In[64]:


T


# In[65]:


#POS Tagging Algorithm - HMM
# computing P(w/t) and storing in T x V matrix
t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))


# In[66]:


# compute word given tag: Emission Probability
def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
    
    return (count_w_given_tag, count_tag)


# In[67]:


# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability

def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# In[68]:


# creating t x t transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)

tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]


# In[69]:


tags_matrix


# In[70]:


# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))


# In[71]:


tags_df


# In[73]:


tags_df.loc['.', :]


# In[74]:


# heatmap of tags matrix
# T(i, j) means P(tag j given tag i)
plt.figure(figsize=(18, 12))
sns.heatmap(tags_df)
plt.show()


# In[75]:


# frequent tags
# filter the df to get P(t2, t1) > 0.5
tags_frequent = tags_df[tags_df>0.5]
plt.figure(figsize=(18, 12))
sns.heatmap(tags_frequent)
plt.show()


# In[76]:


#Viterbi Algorithm


# In[77]:


len(train_tagged_words)


# In[79]:


# Viterbi Heuristic
def Viterbi(words, train_bag = train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))


# In[80]:


#Evaluate a test


# In[81]:


# Running the Viterbi algorithm on a few sample sentences
# since running it on the entire data set will take many hours

random.seed(1234)

# choose random 5 sents
rndom = [random.randint(1,len(test_set)) for x in range(5)]

# list of sents
test_run = [test_set[i] for i in rndom]

# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]

# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]
test_run


# In[82]:


# tagging the test sentences
start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end-start


# In[83]:


print("Time taken in seconds: ", difference)
print(tagged_seq)


# In[84]:


# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 


# In[85]:


accuracy = len(check)/len(tagged_seq)
accuracy


# In[86]:


incorrect_tagged_cases = [[test_run_base[i-1],j] for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0]!=j[1]]


# In[87]:


incorrect_tagged_cases


# In[90]:


## Testing
sentence_test = 'Naũmi na Lũthi Maisyoka, Mbetheleemu'
words = word_tokenize(sentence_test)

start = time.time()
tagged_seq = Viterbi(words)
end = time.time()
difference = end-start


# In[91]:


print(tagged_seq)
print(difference)


# In[ ]:




