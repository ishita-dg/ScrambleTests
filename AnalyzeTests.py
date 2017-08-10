
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns


# In[98]:

id2label = {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
label2id = {'CONTRADICTION': 0, 'NEUTRAL':1, 'ENTAILMENT':2}
tasks = ['adj', 'comp', 'disc', 'iff', 'time', 'verb']
map_name = {'adj' : 'Adjective-Reference binding', 'comp': 'Comparisons', 
            'time': 'Temporal ordering', 'disc': 'Subject-Verb binding',
           'iff': 'Negating a Condition', 'verb': 'Verb Symmetry'}

def loadresults(name, classifier, path, task, verbose = False):
    results = {}
    modelname = name + classifier
    print('\n\n'+'**'*40)
    print("\nLoading test results for {0}...\n".format(modelname))
    print('**'*40 + '\n')

    with open("{0}_labels_{1}.txt".format(path + task, modelname))as f:
        temp = f.readlines()
        results['est'] = [x.strip() for x in temp]
    results['all_est_conf'] = np.loadtxt("testData/{0}_confs_{1}.txt".format(task, modelname))
    with open('{0}_labels.txt'.format(path + task)) as f:
        temp = f.readlines()
        results['true'] = [x.strip() for x in temp]
    
    if verbose:
        with open("{0}_A.txt".format(path + task))as f:
            sents_A = f.readlines()
        with open('{0}_B.txt'.format(path + task)) as f:
            sents_B = f.readlines()
        print("-"*40)
        print("\n Task: ", task, "\n")
        print("-"*40, "\n")
        N = 2
        temp = np.random.randint(0, len(results['true']), N)
        for i in temp:
            print(sents_A[i])
            print(sents_B[i])
            print("True: ", results['true'][i], "\t Estimated: ", results[task]['est'][i] )
            print("\n")

    
    return results


# In[100]:

all_results = {}
names = ['BOW', 'InferSent']
classifiers = [ 'MLP', 'LogReg']
tasks = ['adj', 'comp', 'disc', 'iff', 'time', 'verb']
TEST_PATH = './testData/'

for name in names:
    for classifier in classifiers:
        all_results[name+classifier] = {}
        for task in tasks:
            all_results[name+classifier][task] = loadresults(name, classifier, TEST_PATH, task)


# In[101]:

def plot_hist(data):
    # Create the bar plot
    ax = sns.countplot(
        x="est",
        hue="true",
        order=["CONTRADICTION", "NEUTRAL", "ENTAILMENT"],
        hue_order=["CONTRADICTION", "NEUTRAL", "ENTAILMENT"],
        data=data)
    return plt.gcf(), ax


# In[102]:

for model in all_results:
    plt.figure(figsize=(20, 6))
    for i, task in enumerate(all_results[model]):
        plt.subplot(2,4,i+1)
        a, b = plot_hist(all_results[model][task])
#         lim = max(np.concatenate((np.bincount(all_results[model][task]['est']), np.bincount(all_results[model][task]['true']))))*1.4
#         plt.ylim([0,lim])
        plt.legend(loc='upper right')
        b.set_title('Task = {0}'.format(task))
        lim = max(np.concatenate((np.bincount([label2id[x.strip()] for x in all_results[model][task]['est']]), 
                                  np.bincount([label2id[x.strip()] for x in all_results[model][task]['true']])
                                 )))*1.4
        b.set_ylim([0,lim])
        b.set_xlabel("Estimated label")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('{0}: Histograms of all classifications'.format(model), fontsize = 20)
        



# In[40]:

def condition(t, e, true, est):
    if (true is None):
        return (e == est)
        if (est is None):
            print ("Give at least true or pred")
            return
        else:
            return (t == true)
    elif (est is not None):
        return (e == est & t == true)
    
def show_sents(results, name, classifier, task, true = None, est = None):
    mn = name + classifier
    x = results[mn][task]
    with open("testData/{0}_A.txt".format(task))as f:
        sents_A = f.readlines()
    with open('testData/{0}_B.txt'.format(task)) as f:
        sents_B = f.readlines()
        
    for i, (t,e) in enumerate(zip(x['true'], x['est'])):
        if (condition(t, e, true, est)):
            print(sents_A[i])
            print(sents_B[i])
            print("True: ", t, "\t Estimated: ", e)
            print("\n", "*"*40, "\n")
    


# In[41]:

show_sents(results = all_results, name = "InferSent", classifier = 'LogReg', task = 'iff', est = "NEUTRAL")


# In[42]:

show_sents(results = all_results, name = "InferSent", classifier = 'MLP', task = 'disc', est = "NEUTRAL")


# In[43]:

names = ['BOW', 'InferSent']
classifiers = [ 'MLP', 'LogReg']
tasks = ['verb', 'comp', 'time', 'adj', 'disc', 'iff']
net = 15534.0
for name in names:
    for classifier in classifiers:
        netE = 0
        for task in tasks:
            e = all_results[name+classifier][task]['est']
            t = all_results[name+classifier][task]['true']
            print(name, classifier, task)
            print(np.sum(np.array(e) == np.array(t))*100/float(len(e)))
            print('\n')
            netE += np.sum(np.array(e) == np.array(t))
        netE /= net

        print(netE * 100)
        print('*'*66)
           


# In[97]:

names = ['BOW', 'InferSent']
classifiers = [ 'MLP']
tasks = ['verb', 'comp', 'time', 'adj', 'disc', 'iff']
net = 15534.0
N = 1
plt.figure(figsize=(20, 10))
n_task = 0

for task in tasks:
    n_task +=1 
    print('\nTASK: ', task, "\n\n")
    with open("testData/{0}_A.txt".format(task))as f:
        sents_A = f.readlines()
    with open('testData/{0}_B.txt'.format(task)) as f:
        sents_B = f.readlines()    
   
    tru = np.array(all_results['BOW'+classifier][task]['true'])
    est_B = np.array(all_results['BOW'+classifier][task]['est'])
    est_I = np.array(all_results['InferSent'+classifier][task]['est'])
    filt = np.logical_and(est_B != tru, est_I == tru)
    
    conf_B = np.array([100*all_results['BOW'+classifier][task]['all_est_conf'][i][label2id[j]] for i, j in enumerate(est_B)])
    conf_I = np.array([100*all_results['InferSent'+classifier][task]['all_est_conf'][i][label2id[j]] for i, j in enumerate(est_I)])
    conf_B_intrue = np.array([100*all_results['BOW'+classifier][task]['all_est_conf'][i][label2id[j]] for i, j in enumerate(tru)])
    conf_I_intrue = np.array([100*all_results['InferSent'+classifier][task]['all_est_conf'][i][label2id[j]] for i, j in enumerate(tru)])
    
    conf_B_intrue_nocon = conf_B_intrue[tru != 'CONTRADICTION']
    conf_I_intrue_nocon = conf_I_intrue[tru != 'CONTRADICTION']
    
    diff_con = conf_I - conf_B
    diff_con = conf_B
    order = np.argsort(-diff_con)
    
    disp = 0
    for disp_i in order:
        if disp < N:
            if filt[disp_i]:
                #print(sents_A[disp_i])
                #print(sents_B[disp_i])
                #print("True: ", tru[disp_i], "\t Estimated BOW: ", est_B[disp_i])
                #print("IS conf: ", conf_I[disp_i], "\t BOW conf: ", conf_B[disp_i], "\t BOW conf in true: ", conf_B_intrue[disp_i])
                #print("\n", "*"*40, "\n")

                disp += 1
    plt.subplot(3,2, n_task)
#     plt.hist(conf_B, range = (0,100), normed = True, bins = 30, label = 'BOW', alpha = 0.7)
#     plt.hist(conf_I, range = (0,100), normed = True, bins = 30, label = 'InferSent', alpha = 0.7)
#     plt.hist(conf_B_intrue, range = (0,100), normed = True, bins = 30, label = 'BOW_true', alpha = 0.7)
#     plt.hist(conf_I_intrue, range = (0,100), normed = True, bins = 30, label = 'InferSent_true', alpha = 0.7)
    plt.hist(conf_B_intrue_nocon, range = (0,100), normed = True, bins = 30, label = 'BOW_true_nocon', alpha = 0.7)
    plt.hist(conf_I_intrue_nocon, range = (0,100), normed = True, bins = 30, label = 'InferSent_true_nocon', alpha = 0.7)
    
    plt.legend(loc='upper left')
    plt.title('Task: {0}'.format(map_name[task]))
    

plt.suptitle('Confidences - MLP')        
plt.show()   


# In[ ]:

names = ['BOW', 'InferSent']
classifiers = [ 'MLP']
tasks = ['verb', 'comp', 'time', 'adj', 'disc', 'iff']
net = 15534.0
N = 1
plt.figure(figsize=(20, 10))
n_task = 0

for task in tasks:
    n_task +=1 
    print('\nTASK: ', task, "\n\n")
    with open("testData/{0}_A.txt".format(task))as f:
        sents_A = f.readlines()
    with open('testData/{0}_B.txt'.format(task)) as f:
        sents_B = f.readlines()    
   
    tru = np.array(all_results['BOW'+classifier][task]['true'])
    est_B = np.array(all_results['BOW'+classifier][task]['est'])
    est_I = np.array(all_results['InferSent'+classifier][task]['est'])
    filt = np.logical_and(est_B != tru, est_I == tru)
    
    conf_B = np.array([100*all_results['BOW'+classifier][task]['all_est_conf'][i][label2id[j]] for i, j in enumerate(est_B)])
    conf_I = np.array([100*all_results['InferSent'+classifier][task]['all_est_conf'][i][label2id[j]] for i, j in enumerate(est_I)])
    conf_B_intrue = np.array([100*all_results['BOW'+classifier][task]['all_est_conf'][i][label2id[j]] for i, j in enumerate(tru)])
    conf_I_intrue = np.array([100*all_results['InferSent'+classifier][task]['all_est_conf'][i][label2id[j]] for i, j in enumerate(tru)])
    
    conf_B_intrue_nocon = conf_B_intrue[tru != 'CONTRADICTION']
    conf_I_intrue_nocon = conf_I_intrue[tru != 'CONTRADICTION']
    
    diff_con = conf_I - conf_B
    diff_con = conf_B
    order = np.argsort(-diff_con)
    
    disp = 0
    for disp_i in order:
        if disp < N:
            if filt[disp_i]:
                #print(sents_A[disp_i])
                #print(sents_B[disp_i])
                #print("True: ", tru[disp_i], "\t Estimated BOW: ", est_B[disp_i])
                #print("IS conf: ", conf_I[disp_i], "\t BOW conf: ", conf_B[disp_i], "\t BOW conf in true: ", conf_B_intrue[disp_i])
                #print("\n", "*"*40, "\n")

                disp += 1
    plt.subplot(3,2, n_task)
#     plt.hist(conf_B, range = (0,100), normed = True, bins = 30, label = 'BOW', alpha = 0.7)
#     plt.hist(conf_I, range = (0,100), normed = True, bins = 30, label = 'InferSent', alpha = 0.7)
#     plt.hist(conf_B_intrue, range = (0,100), normed = True, bins = 30, label = 'BOW_true', alpha = 0.7)
#     plt.hist(conf_I_intrue, range = (0,100), normed = True, bins = 30, label = 'InferSent_true', alpha = 0.7)
    plt.hist(conf_B_intrue_nocon, range = (0,100), normed = True, bins = 30, label = 'BOW_true_nocon', alpha = 0.7)
    plt.hist(conf_I_intrue_nocon, range = (0,100), normed = True, bins = 30, label = 'InferSent_true_nocon', alpha = 0.7)
    
    plt.legend(loc='upper left')
    plt.title('Task: {0}'.format(map_name[task]))
    

plt.suptitle('Confidences - MLP')        
plt.show()   

