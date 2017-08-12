
# coding: utf-8

# In[1]:

from random import randint
import matplotlib

import numpy as np
import torch
import pickle
import regrFuncs as rF

def predict(v1, v2, reg):
    feats = rF.featurize(v1, v2)
    probs = reg.predict_proba(feats)
    label = np.array([int(x) for x in reg.predict(feats).flatten()])
    return label, probs.tolist()


def runtests(name, classifier, model, tasks, outpaths, label2id):
    modelname = name + classifier
    print('\n\n'+'**'*40)
    print("\nRunning tests for {0}...\n".format(modelname))
    print('**'*40 + '\n')
    regressor = pickle.load(open('{0}{1}'.format(outpaths['REGR_MODEL_PATH'], modelname), 'rb'))
    for task in tasks:
        with open('{0}s1.{1}'.format(outpaths['TEST_DATA_PATH'], task)) as f:
            X_A = f.readlines()
        with open('{0}s2.{1}'.format(outpaths['TEST_DATA_PATH'], task)) as f:
            X_B = f.readlines()
        try:
            true = np.loadtxt('{0}labels.{1}'.format(outpaths['TEST_DATA_PATH'], task), 
                dtype = int)
        except ValueError:
            with open('{0}labels.{1}'.format(outpaths['TEST_DATA_PATH'], task)) as f:
                true = f.readlines()
                true = [label2id[x.strip().upper()] for x in true]

        
        embedA = rF.embed(model, X_A, 1, name)
        embedB = rF.embed(model, X_B, 1, name)
        
        labels, confs = predict(embedA, embedB, regressor)    
              
        np.savetxt('{0}{1}_labels_{2}'.format(outpaths['TEST_OUT_PATH'], task, modelname),
            labels,
            fmt = '%i')

        np.savetxt('{0}{1}_confs_{2}'.format(outpaths['TEST_OUT_PATH'], task, modelname), 
            confs)
        
        rights = (labels == true)
        acc = sum(rights)*100.0/len(rights)
        
        counts = np.bincount(true, minlength = 3)*100.0/len(rights)
        counts = np.array([round(x,2) for x in counts])
        
        print("\nTask: {0}, Contradiction: {1}, Neutral: {2}, "
            "Entailment: {3}".format(task, counts[0], counts[1], counts[2]))
        print("Accuracy: {0}".format(round(acc,3)))

    return 
# open('{0}_X_B_self.txt'.format(modelname), 'w').write("\n".join([str(label2id[x]) for x in labels]))
# np.savetxt('{0}_X_B_self_conf.txt'.format(modelname), confs)

