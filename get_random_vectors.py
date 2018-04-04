# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet
#from IPython import embed
import loadData as lD
from IPython import embed
from tqdm import tqdm

GLOVE_PATH = "dataset/GloVe/glove.840B.300d.txt"


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/true/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='newdir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model-retrain.pickle')

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

parser.add_argument("--toy", type=bool, default=False, help="toy")


params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""

toy = params.toy

"""
# toy SNLI provided, download the rest
GLOVE_PATH = './Downloads/glove.840B.300d.txt'
MODEL_PATH = './Downloads/infersent.allnli.pickle'
REGR_MODEL_PATH = './models/'
# If not None, where you want SNLI embeddings to be stored (WARNING: high memory)
EMBED_STORE = None
TEST_OUT_PATH = './regout/'

if toy:
    DATA_PATH = './Downloads/SNLI/toy/'
    REGR_MODEL_PATH = REGR_MODEL_PATH + 'TOY'
    EMBED_STORE = None
    TEST_OUT_PATH = TEST_OUT_PATH + 'TOY'
    TEST_DATA_PATH = './testData/toy/'
else:
    DATA_PATH = './Downloads/SNLI/true/'
    TEST_DATA_PATH = './testData/true/'

outpaths = {'REGR_MODEL_PATH': REGR_MODEL_PATH, 'TEST_OUT_PATH': TEST_OUT_PATH, 'TEST_DATA_PATH' : TEST_DATA_PATH}


"""
TEST_DATA_PATH = './testData/true/'
TEST_OUT_PATH = './regout/'
REGR_MODEL_PATH = './models/'
outpaths = {'REGR_MODEL_PATH': REGR_MODEL_PATH, 'TEST_OUT_PATH': TEST_OUT_PATH, 'TEST_DATA_PATH' : TEST_DATA_PATH}

id2label = {0:'CONTRADICTION', 1:'NEUTRAL', 2:'ENTAILMENT'}
label2id = {'CONTRADICTION': 0, 'NEUTRAL':1, 'ENTAILMENT':2}

#tasks = ['adjr', 'comp', 'ncon', 'subjv', 'temp', 'verb']
#tasks = [f[7:] for f in os.listdir(outpaths['TEST_DATA_PATH']) if 'label' in f]
tasks = ["comp_ml_long", "comp_ml_short"]

# Load scramble data in same format as snli (suitable for training)
scramble_data_path = './testData/toy/' if toy else './testData/true/'
scramble_train = lD.load_scramble_all("./testData/new/train/", label2id, tasks, one_group=True)
scramble_valid = lD.load_scramble_all("./testData/new/valid/", label2id, tasks, one_group=True)
scramble_test = lD.load_scramble_all("./testData/new/test/", label2id, tasks, one_group=True)

""" Convert Compositional Dataset format to SNLI dataset """
def convert_format(data, limit=5000):
    new_data = {}
    new_data["s1"] = data["X_A"][:limit]
    new_data["s2"] = data["X_B"][:limit]
    new_data["label"] = np.array(data["y"][:limit])

    return new_data

def merge_data(data1, data2):
    data1["s1"].extend(data2["s1"])
    data1["s2"].extend(data2["s2"])
    data1["label"] = np.concatenate((data1["label"], data2["label"]), axis=0)
    return data1
train_snli, valid_snli, test_snli = get_nli(params.nlipath)
train_comp = convert_format(scramble_train)
valid_comp = convert_format(scramble_valid)
test_comp = convert_format(scramble_test)

train = merge_data(train_comp, train_snli)

word_vec = build_vocab(train_snli['s1'] + train_snli['s2'] +
                       valid_snli['s1'] + valid_snli['s2'] +
                       test_snli['s1'] + test_snli['s2'] +
                       train_comp['s1'] + train_comp['s2'] +
                       valid_comp['s1'] + valid_comp['s2'] +
                       test_comp['s1'] + test_comp['s2'], GLOVE_PATH)

vocab_size = len(word_vec)
dim = len(word_vec[","])
words = list(word_vec)
def avg_100(n=100):
    ids = random.sample(range(vocab_size), n)
    new_vec = np.zeros((dim))
    for i in ids:
       new_vec = new_vec + np.array(word_vec[words[i]])
    new_vec /= 1.0 * n
    return new_vec.tolist()


import spacy
nlp = spacy.load('en')
noun_words = []
for word in tqdm(words):
    pos = nlp(word)[0].pos_
    if pos == "NOUN":
        noun_words.append(word)
noun_size = len(noun_words)

def avg_noun_100(n=100):
    ids = random.sample(range(noun_size), n)
    new_vec = np.zeros((dim))
    for i in ids:
        new_vec = new_vec + np.array(word_vec[noun_words[i]])
    new_vec /= 1.0 * n
    return new_vec.tolist()


def random_word_vec():  # noun
    ids = random.sample(range(noun_size), 1)
    idx = ids[0]
    new_vec = word_vec[noun_words[idx]]
    return new_vec.tolist()

def print_vec(vec):
    s = ["%.3lf" % x for x in vec]
    return " ".join(s)

f = open("random_noun.vectors", "w")
for i in range(500):
    new_vec = avg_noun_100()
    f.write("<rand_noun_%d> %s\n" % (i, print_vec(new_vec)))
f.close()
    
