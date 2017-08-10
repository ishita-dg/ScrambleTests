
# coding: utf-8

import numpy as np

def loadsens(path):
    out = []
    with open(path, 'r') as f:
        for line in f:
            text = line.strip()
            out.append(text)
    return out

def loadlabels(path, label2id):
    out = []
    with open(path, 'r') as f:
        for line in f:
            text = line.strip()
            out.append(label2id[text.upper()])
    return out


def loadSNLI(path, label2id):
    snli_data = {'train':{}, 'dev':{}, 'test':{}}

    for key in snli_data:  
    
        print('Loading data for {0}'.format(key))
        
        snli_data[key] = {'X_A': loadsens(path + 's1.' + key),
                          'X_B': loadsens(path + 's2.' + key), 
                          'y': loadlabels(path + 'labels.' + key, label2id)}

        # Sort to reduce padding
        sorted_corpus = sorted(zip(snli_data[key]['X_A'],
                                   snli_data[key]['X_B'],
                                   snli_data[key]['y']),
                               key=lambda z:(len(z[0]), len(z[1]), z[2]))
        
        snli_data[key]['X_A'] = [x for (x,y,z) in sorted_corpus]
        snli_data[key]['X_B'] = [y for (x,y,z) in sorted_corpus]
        snli_data[key]['y'] = [z for (x,y,z) in sorted_corpus]
    
    return snli_data





