import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
import torch
import nltk
nltk.download("punkt")
import torch
import pickle
import regrFuncs as rF
import testFuncs as tF
from tqdm import tqdm

import os, sys

GLOVE_PATH = './Downloads/glove.840B.300d.txt'
MODEL_PATH = './Downloads/infersent.allnli.pickle'
REGR_MODEL_PATH = './models/'
EMBED_STORE = None
TEST_OUT_PATH = './regout/'
DATA_PATH = './Downloads/SNLI/true/'

outpaths = {'REGR_MODEL_PATH': REGR_MODEL_PATH, 'TEST_OUT_PATH': TEST_OUT_PATH}


id2label = {0:'CONTRADICTION', 1:'NEUTRAL', 2:'ENTAILMENT'}
label2id = {'CONTRADICTION': 0, 'NEUTRAL':1, 'ENTAILMENT':2}

print "MODEL_PATH=", MODEL_PATH
print "cwd=", os.getcwd()

model = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
model.use_cuda = False
model.set_glove_path(GLOVE_PATH)
model.build_vocab_k_words(K=100000)

names = ['InferSent', 'BOW']
classifiers = [ 'LogReg']
all_regs = {}
for name in names:
    for classifier in classifiers:
        all_regs[name+classifier] = pickle.load(open('{0}{1}'.format(outpaths['REGR_MODEL_PATH'], name+classifier), 'rb'))

def print_preds(sent_a, sent_b, verbose = True, names = names, classifiers = classifiers):
    vals = {}
    for name in names:
        for classifier in classifiers:
            A, B = rF.embed(model, sent_a, 1, name), rF.embed(model, sent_b, 1, name)
            pred, conf = tF.predict(A, B, all_regs[name+classifier])
            if verbose:
                print('*'*20)
                print(name, classifier)
                print('*'*20, '\n')
            vals[name + classifier] = {}
            vals[name + classifier]['pred'] = []
            vals[name + classifier]['conf'] = []
            for i in range(len(A)):
                if verbose:
                    print('A: ', sent_a[i], '\t B: ', sent_b[i])
                    print(id2label[pred[i]], conf[i][pred[i]]*100)
                    print('\n')
                vals[name + classifier]['pred'].append(id2label[pred[i]])
                vals[name + classifier]['conf'].append(conf[i][pred[i]]*100)
            vals[name + classifier]['pred'] = np.array(vals[name + classifier]['pred'])
            vals[name + classifier]['conf'] = np.array(vals[name + classifier]['conf'])
            if verbose:
                print('\n\n')
    return vals

def external_visualize(model, sent, tokenize=True, output_y=False):
    if tokenize: from nltk.tokenize import word_tokenize
        
    sent = sent.split() if not tokenize else word_tokenize(sent)
    sent = [['<s>'] + [word for word in sent if word in model.word_vec] + ['</s>']]

    if ' '.join(sent[0]) == '<s> </s>':
        import warnings
        warnings.warn('No words in "{0}" have glove vectors. Replacing by "<s> </s>"..'.format(sent))
    batch = torch.autograd.Variable(model.get_batch(sent), volatile=True)
        
    if model.use_cuda:
        batch = batch.cuda()
    output = model.enc_lstm(batch)[0]
    output, idxs = torch.max(output, 0)
    #output, idxs = output.squeeze(), idxs.squeeze()
    idxs = idxs.data.cpu().numpy()
    argmaxs = [np.sum((idxs==k)) for k in range(len(sent[0]))]
        
        # visualize model
    import matplotlib.pyplot as plt
    x = range(len(sent[0]))
    y = [100.0*n/np.sum(argmaxs) for n in argmaxs]
    if output_y:
        return output, idxs, y
    fig = plt.figure()
    plt.xticks(x, sent[0], rotation=45)
    plt.bar(x, y)
    plt.ylabel('%')
    plt.title('Visualisation of words importance')
    
    plt.show()
        
    return output, idxs

from sets import Set
from tqdm import tqdm
def weighted_overlap_percentage(num_sent, s1, s2, labels, valid_labels=[], max_sent=100):
    overlap = 0
    total = 0
    for i in tqdm(range(num_sent)):
        if len(valid_labels) != 0 and labels[i] not in valid_labels:
            continue
        max_sent -= 1
        if max_sent == 0:
            return overlap / total
        s1_set = Set(s1[i])
        s2_set = Set(s2[i])
        cur_overlap = 0
        cur_total = 0
        _, _, s1_imp = external_visualize(model, " ".join(s1[i]), tokenize=True, output_y=True)
        _, _, s2_imp = external_visualize(model, " ".join(s2[i]), tokenize=True, output_y=True)
        for p in range(len(s1[i])):
            word = s1[i][p]
            if word in s2_set:
                cur_overlap += s1_imp[p]
            cur_total += s1_imp[p]
        for p in range(len(s2[i])):
            word = s2[i][p]
            if word in s1_set:
                cur_overlap += s2_imp[p]
            cur_total += s2_imp[p]
        cur_overlap /= 1.0 * cur_total
        overlap += cur_overlap
        total += 1
    return overlap / total
            
def read_data(s1, s2, labels):
    f1 = open(s1)
    f2 = open(s2)
    lab = open(labels)
    sent1 = []
    for line in f1:
        sent1.append(line.replace(" \n", "").split(" "))
    sent2 = []
    for line in f2:
        sent2.append(line.replace(" \n", "").split(" "))
    labs = []
    for line in lab:
        labs.append(line.replace("\n", ""))
    f1.close()
    f2.close()
    lab.close()
    return sent1, sent2, labs

s1, s2, labels = read_data("s1.train", "s2.train", "labels.train")
num_sent = len(s1)

"""How many contradictory pair of sentences has antonyms + print out importance of those antonyms"""
import nltk
from nltk.corpus import wordnet
def get_antonym_set(word):
    antonyms = Set([])
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.add(l.antonyms()[0].name())
    return antonyms

def antonym_analysis(num_sent, s1, s2, labels, valid_labels=[], imp_sample=10):
    
    sum_antonym = 0
    counter = 0
    total = 0
    for i in range(num_sent):
        if len(valid_labels) != 0 and labels[i] not in valid_labels:
            continue
        cur_antonym = 0
        for word in s1[i]:
            antonym = get_antonym_set(word)
            if len(Set(s2[i]) & antonym) != 0:
                cur_antonym += 1
        sum_antonym += cur_antonym
        counter += int(cur_antonym > 0)
        total += 1
        
        if imp_sample != 0:
            print "label=", labels[i]
            model.visualize(" ".join(s1[i]), tokenize=True)
            model.visualize(" ".join(s2[i]), tokenize=True)
            _, _, y1 = external_visualize(model," ".join(s1[i]), tokenize=True,output_y=True)
            _, _, y2 = external_visualize(model," ".join(s2[i]), tokenize=True,output_y=True)
            imp = 0
            for idx, word in enumerate(s1[i]):
                antonym = get_antonym_set(word)
                if len(Set(s2[i]) & antonym) != 0:
                    imp += y1[idx]
            imp /= cur_antonym
            print "avg imp of antonyms (in s1): %.3lf" % (imp)
            imp_sample -= 1
    print "avg antonym # for given type sentence: %d" % (sum_antonym * 1.0 / total)
    print "percantage of given type sentence to have antonyms: %d/%d (%.5lf)" % (counter, total, 1.0 * counter / total)

antonym_analysis(num_sent, s1, s2, labels, valid_labels=["contradiction"])
        