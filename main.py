
# coding: utf-8

# # Training a classifier from InferSent embeddings to SNLI
# 
# This is how the embeddings were trained end-to-end as well, but we just try with the given embeddings and SNLI training set.

# In[1]:

# %load_ext autoreload
# %autoreload 2

from random import randint
import matplotlib

import numpy as np
import torch
import nltk
nltk.download("punkt")
import torch
import pickle
import classifier as cl
import regrFuncs as rF
import testFuncs as tF
import loadData as lD
import os


toy = True

# toy SNLI provided, download the rest
GLOVE_PATH = './Downloads/glove.840B.300d.txt'
MODEL_PATH = './Downloads/infersent.allnli.pickle'
if toy:
    DATA_PATH = './Downloads/SNLI/toy/'
else:
    DATA_PATH = './Downloads/SNLI/true/'

# Where you want regression models to be stored (pre-trained provided)
REGR_MODEL_PATH = './models/'
# If not None, where you want SNLI embeddings to be stored (WARNING: high memory)
EMBED_STORE = None
TEST_OUT_PATH = './regout/'


# toy embeddings neVer stored, and the stored models are marked
if toy:
    REGR_MODEL_PATH = REGR_MODEL_PATH + 'TOY'
    EMBED_STORE = None
    TEST_OUT_PATH = TEST_OUT_PATH + 'TOY'

outpaths = {'REGR_MODEL_PATH': REGR_MODEL_PATH, 'TEST_OUT_PATH': TEST_OUT_PATH}


id2label = {0:'CONTRADICTION', 1:'NEUTRAL', 2:'ENTAILMENT'}
label2id = {'CONTRADICTION': 0, 'NEUTRAL':1, 'ENTAILMENT':2}


model = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
model.use_cuda = False
model.set_glove_path(GLOVE_PATH)
if toy:
    model.build_vocab_k_words(K=100)
else:
    model.build_vocab_k_words(K=100000)

batch_size = 64

snli_data = lD.loadSNLI(DATA_PATH, label2id)

# Load scramble data in same format as snli (suitable for training)
scramble_data_path = './testData/toy/' if toy else './testData/true/'
scramble_data = lD.load_scramble_all(scramble_data_path, label2id)

# Create combined dataset of SNLI + scramble data
combined_data = lD.sort_group(lD.merge_groups([scramble_data, snli_data]))

# Select which data to train on:
training_data = snli_data # combined_data, scramble_data

names = ['InferSent','BOW']
classifiers = ['LogReg', 'MLP']

def allClassifiersExist(name):
    flag = True
    for classifier in classifiers:
        flag *= os.path.exists(REGR_MODEL_PATH + name + classifier)
    return flag

# Train regressions
for name in names:
    if (not allClassifiersExist(name)):
        embeddings = rF.create_embed(model, training_data, batch_size, name, EMBED_STORE)
    for classifier in classifiers:
        if (not os.path.exists(REGR_MODEL_PATH + name + classifier)):
            rF.trainreg(embeddings, training_data, classifier, name, outpaths)


# Test 
# Scrambletest tasks
tasks = ['adjr', 'comp', 'ncon', 'subjv', 'temp', 'verb']
if toy:
    outpaths['TEST_DATA_PATH'] = './testData/toy/'
else:
    outpaths['TEST_DATA_PATH'] = './testData/true/'


for name in names:
    for classifier in classifiers:
        tF.runtests(name, classifier, model, tasks, outpaths)


# Retest the trained regressions
tasks = ['test']
outpaths['TEST_DATA_PATH'] = DATA_PATH

for name in names:
    for classifier in classifiers:
        tF.runtests(name, classifier, model, tasks, outpaths, label2id)

