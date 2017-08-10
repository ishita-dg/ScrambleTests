from random import randint
import matplotlib

import numpy as np
import torch
import nltk
import torch
import pickle
import classifier as cl
import os
from sklearn.linear_model import LogisticRegression

def embed(model, batch, batch_size, name):
    if (name == 'BOW'):
        embeddings = []
        batch = [sent if sent!=[] else ['.'] for sent in batch]
        for sent in batch:
            sentvec = []
            for word in sent:
                if word in model.word_vec:
                    sentvec.append(model.word_vec[word])
            if not sentvec:
                sentvec.append(model.word_vec['.'])
            sentvec = np.mean(sentvec, 0)
            embeddings.append(sentvec)
    
        embeddings = np.vstack(embeddings)
        return embeddings
    elif (name == 'InferSent'):
        embeddings = model.encode(batch, bsize = batch_size, tokenize = False)
        return embeddings
    else:
        raise NameError('Model not included')
    
def create_embed(model, data, batch_size, name, EMBED_STORE = None):
    print('\nStart embedding for {0}\n'.format(name))
    snli_embed = {'train':{}, 'dev':{}, 'test':{}} 
    for key in snli_embed:
        period = round(len(data[key]['y'])/10)
        print('Computing embedding for {0}'.format(key))
        fac = int(len(data[key]['y'])/(batch_size))
        for txt_type in ['X_A', 'X_B']:
            if (EMBED_STORE is not None):
                fname = EMBED_STORE + name + key + txt_type
                if os.path.exists(fname):
                    snli_embed[key][txt_type] = np.loadtxt(fname)
            else:
                snli_embed[key][txt_type] = []
                for ii in range(0, len(data[key]['y']), batch_size):
                    batch = data[key][txt_type][ii:ii + batch_size]
                    embeddings = embed(model, batch, batch_size, name)
                    snli_embed[key][txt_type].append(embeddings)
                    if (ii/batch_size)%(fac) == 0:
                        print("PROGRESS (encoding): {0}%".format(100.0 * ii /len(data[key]['y'])))
                snli_embed[key][txt_type] = np.vstack(snli_embed[key][txt_type])
                if (EMBED_STORE is not None) :
                    np.savetxt(fname, snli_embed[key][txt_type])
                

        print('Computed {0} embeddings\n'.format(key))
    
    return(snli_embed)


# In[34]:

class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, X, y, config, outpaths, name):
        self.X = X
        self.y = y
        self.nclasses = config['nclasses']
        self.featdim = self.X['train'].shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier = config['classifier']
        self.nhid = config['nhid']
        self.name = name
        self.outpaths = outpaths
        self.cudaEfficient = False if 'cudaEfficient' not in config else config['cudaEfficient']
        self.modelname = 'sklearn-LogReg' if not config['usepytorch'] else 'pytorch-' + config['classifier']
        self.nepoches = None if 'nepoches' not in config else config['nepoches']
        self.maxepoch = None if 'maxepoch' not in config else config['maxepoch']
        self.noreg = False if 'noreg' not in config else config['noreg']
        
    def run(self):
        print('Training {0}, {1} with standard validation..'.format(self.modelname, self.name))
        regs = [10**t for t in range(-5,-1)] if self.usepytorch else [2**t for t in range(-2,4,1)]
        if self.noreg : regs=[0.]
        scores = []
        for reg in regs:
            if self.usepytorch:
                if self.classifier == 'LogReg':
                    clf = cl.LogReg(inputdim=self.featdim, nclasses=self.nclasses, l2reg=reg, 
		                    seed=self.seed, cudaEfficient=self.cudaEfficient)
                elif self.classifier == 'MLP':
                    clf = cl.MLP(inputdim=self.featdim, hiddendim=self.nhid, nclasses=self.nclasses,                            
		                 l2reg=reg, seed=self.seed, cudaEfficient=self.cudaEfficient)
                # small hack : MultiNLI/SNLI specific
                if self.nepoches: clf.nepoches = self.nepoches
                if self.maxepoch: clf.maxepoch = self.maxepoch
                clf.fit(self.X['train'], self.y['train'], validation_data=(self.X['valid'], self.y['valid']))
            else:
                clf = LogisticRegression(C=reg, random_state=self.seed)
                clf.fit(self.X['train'], self.y['train'])
            scores.append(round(100*clf.score(self.X['valid'], self.y['valid']),2))
        print([('reg:'+str(regs[idx]), scores[idx]) for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        print('Validation : best param found is reg = {0} with score {1}'.format(optreg, devaccuracy))               
        print('Evaluating...')
        if self.usepytorch:
            if self.classifier == 'LogReg':
                clf = cl.LogReg(inputdim = self.featdim, nclasses=self.nclasses, l2reg=optreg,                            
		                seed=self.seed, cudaEfficient=self.cudaEfficient)
            elif self.classifier == 'MLP':
                clf = cl.MLP(inputdim = self.featdim, hiddendim=self.nhid, nclasses=self.nclasses,                          
		             l2reg=optreg, seed=self.seed, cudaEfficient=self.cudaEfficient)
            # small hack : MultiNLI/SNLI specific
            if self.nepoches: clf.nepoches = self.nepoches
            if self.maxepoch: clf.maxepoch = self.maxepoch
            devacc = clf.fit(self.X['train'], self.y['train'], validation_data=(self.X['valid'], self.y['valid']))
        else:
            # changing solver to multinomial
            clf = LogisticRegression(C=optreg, random_state=self.seed, 
                                     solver = 'sag', multi_class = 'multinomial')
            clf.fit(self.X['train'], self.y['train'])
        
        
        fname = self.outpaths['REGR_MODEL_PATH'] + self.name + self.classifier
        pickle.dump(clf, open(fname, 'wb'))
        clf = pickle.load(open(fname, 'rb'))

        prediction = [x for x in clf.predict(self.X['test']).flatten()]
        confidence = clf.predict_proba(self.X['test']).tolist()
        np.savetxt(self.outpaths['TEST_OUT_PATH'] + 'test_labels_' 
            + self.name + self.classifier, 
            prediction,
            fmt = '%i')
       	np.savetxt(self.outpaths['TEST_OUT_PATH'] + 'test_confs_' 
            + self.name + self.classifier, 
            confidence)
        
        testaccuracy = clf.score(self.X['test'], self.y['test'])
        testaccuracy = round(100*testaccuracy, 2)
        return devaccuracy, testaccuracy
    


# In[35]:

def featurize(v1, v2):
    return np.c_[v1, v2, np.abs(v1 - v2), v1*v2]

def trainreg (embed, data, classifier, name, outpaths):
    # Train
    trainF = featurize(embed['train']['X_A'], embed['train']['X_B'])
    trainY = np.array(data['train']['y'])
    
    # Dev
    devF = featurize(embed['dev']['X_A'], embed['dev']['X_B'])
    devY = np.array(data['dev']['y'])
    
    # Test
    testF = featurize(embed['test']['X_A'], embed['test']['X_B'])
    testY = np.array(data['test']['y'])
    
    
    config_classifier = {'nclasses':3, 'seed':1111, 'usepytorch':True,
                         'classifier': classifier, 'nhid': 512}
    
    clf = SplitClassifier(X={'train':trainF, 'valid':devF, 'test':testF},
                          y={'train':trainY, 'valid':devY, 'test':testY},
                          config=config_classifier, outpaths = outpaths,
                          name = name)
    
    devacc, testacc = clf.run()
    print('\nDev acc : {0} Test acc : {1} for SNLI entailment\n'.format(devacc, testacc))
    
    return

