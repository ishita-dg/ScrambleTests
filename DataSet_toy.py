
# coding: utf-8

# # Moving from words to sentences
# 
# ### What is the most basic thing we want to be able to do with more than just word-level information?
# 
# We propose that natural language inference is a good domain to test for whether relational information between words is being used. Humans are good at it and give predictable answers to these questions, and they require concrete and tangible realtional information between words to get to the right answer.
# 
# ### Datasets that require increasing amounts of non-local information
# 
# Several tasks now are interested in sentence representations that go beyond bag-of-words. Sentiment analysis and paraphrase datasets go slightly above, but a lot of the performance of most models on these comes from word-level information. While sentence representations do outperform BOW on these, it is unclear exactly where they have improved.
# 
# Natural language inference is a useful domain in which we can propose challenges that require increasingly complex compositionality and are therefore more diagnostic for what is being learnt and what isn't.
# 
# We present this set of datasets for natural language inference that humans perform predictably well on, and are impossible to capture from word-level information. 
# 
# ### Choosing a vocabulary
# We chose the SNLI dataset vocabulary, so that we could benchmark on the InferSent model that was trained end-to-end on natural language inference with this dataset. This assumes GloVe word embeddings.
# 
# NOTE : I haven't actually checked if these examples are within the vocab, but that's easy to do.
# 

# In[1]:

import numpy as np
id2label = {0:'CONTRADICTION', 1:'NEUTRAL', 2:'ENTAILMENT'}
label2id = {'CONTRADICTION': 0, 'NEUTRAL':1, 'ENTAILMENT':2}


# ### Kinds of examples:
# 
# #### Requiring word-level information regarding its symmetry
# 
# A. **Symmetric vs non-symmetric verbs (over subject-object):**

# In[2]:

# Insensitive to tense for now (?)

v_ents = ['meets']
v_cons = ['overtakes']
# also 'causes'
v_neus = ['watches']

# Perhaps I should select from the most commonly used verbs...? And get rid of preposition phrase based ones....? 
# Overlaps with comparatives slightly otherwise

nps = ["the fat man", "the old man holding an umbrella"]

sents_A = []
sents_B = []
labels = []

vs = {"ENTAILMENT": v_ents,
     "CONTRADICTION": v_cons,
     "NEUTRAL": v_neus}

for np1 in nps:
    for np2 in nps:
        for key in vs:
            for v in vs[key]:
                if (np1 != np2):
                    sents_A.append(np1 + " " + v + " " + np2 + ' . ')
                    sents_B.append(np2 + " " + v + " " + np1 + ' . ')
                    labels.append(key)
                    
#                     # self-rep
#                     sents_A.append(np1 + " " + v + " " + np2 + ' . ')
#                     sents_B.append(np1 + " " + v + " " + np2 + ' . ')
#                     labels.append('ENTAILMENT')
                    



open("testData/toy/s1.verb", 'w').write("\n".join([str(x) for x in sents_A]))
open("testData/toy/s2.verb", 'w').write("\n".join([str(x) for x in sents_B]))
np.savetxt("testData/toy/labels.verb", [label2id[x] for x in labels], fmt='%i')
    
#FUTURE
# Give people on Mturk a noun phrase + verb and ask them to fill it out with a phrase
# such that the converse makes sense (for asym_v's)
# We ca nbootstrap noun phrases that people give us for more sentences


# B. **Temporal ordering**

# In[3]:

tws = ["after"]
infs = ["CONTRADICTION"]
vps = ['sat down', 'walked in']
nps = ["the woman in the black shirt", "the boy in the red shorts"]

sents_A = []
sents_B = []
labels = []

for vp1 in vps:
    for vp2 in vps:
        for np1 in nps:
            for np2 in nps:
                for w, inf in zip(tws, infs):
                    if ((np1 != np2) & (vp1 != vp2)):
                        sents_A.append(np1 + " " + vp1 + " " + w + " " + np2 + ' ' + vp2 + ' . ')
                        sents_B.append(np2 + " " + vp2 + " " + w + " " + np1 + ' ' + vp1 + ' . ')
                        labels.append(inf)
                        
#                         # equalize numbers of each type
#                         u = np.random.uniform()
#                         if (u > 0.5):
#                             sents_A.append(np1 + " " + vp1 + " " + w + " " + np2 + ' ' + vp2 + ' . ')
#                             sents_B.append(np2 + " " + vp1 + " " + w + " " + np1 + ' ' + vp2 + ' . ')
#                             labels.append("NEUTRAL")
                
#                 # self-rep
#                 sents_A.append(np1 + " " + w + " " + np2 + ' . ')
#                 sents_B.append(np1 + " " + w + " " + np2 + ' . ')
#                 labels.append('ENTAILMENT')
                    



open("testData/toy/s1.temp", 'w').write("\n".join([str(x) for x in sents_A]))
open("testData/toy/s2.temp", 'w').write("\n".join([str(x) for x in sents_B]))
np.savetxt("testData/toy/labels.temp", [label2id[x] for x in labels], fmt = '%i')

# FUTURE
# Give people on Mturk a noun phrase + "after" and ask them to fill it out, 
# permute order and include with "before, as and while"


# #### Requiring bi-gram compositionality
# 
# A. Modifiers (adjectives)

# In[4]:

vps = ['meets']
# vps = ['meets', 'resembles', 'watches', 'ignores', 'hits']

adjs_temp = {'pos': ['tall', 'cheerful'],
          'neg': ['short', 'grumpy']}
# adjs = {'pos': ['tall', 'big', 'fat'],
#           'neg': ['short', 'small', 'thin']}

adjs = {}
adjs['pos'] = adjs_temp['pos'] + adjs_temp['neg']
adjs['neg'] = adjs_temp['neg'] + adjs_temp['pos']

nps = ["old man holding an umbrella",
       "girl", "boy"]


sents_A = []
sents_B = []
labels = []

for vp in vps:
    for np1 in nps:
        for np2 in nps:
            if (np1 != np2):
                for p, n in zip(adjs['pos'], adjs['neg']):
                    
                    sents_A.append('The ' +  np1 + ' who is ' + p + ', ' + vp + ' the ' + np2 + ' who is ' + n + ' . ')
                    sents_B.append('The ' +  np1 + ', ' + vp + ' the ' + np2 + ' who is ' + n + ' . ')
                    labels.append('ENTAILMENT')
                    
                    sents_A.append('The ' +  np1 + ' who is ' + p + ', ' + vp + ' the ' + np2 + ' who is ' + n + ' . ')
                    sents_B.append('The ' + np1 + ' who is ' + n + ', '+ vp + ' the ' + np2 + ' . ')
                    labels.append('CONTRADICTION')
                
                    
        
        
open("testData/toy/s1.adjr", 'w').write("\n".join([str(x) for x in sents_A]))
open("testData/toy/s2.adjr", 'w').write("\n".join([str(x) for x in sents_B]))
np.savetxt("testData/toy/labels.adjr", [label2id[x] for x in labels], fmt='%i')


# FUTURE
# Find all sentences in s2 from multiNLI that have two non consecutive adjectives in them and swap them, 
# check with Mturk for label - because they won't be opposites like here


# B. Modifiers that negate - if and only if

# In[5]:

connec = ['when', 'if']
phe = {'pos': ['it rains', 'there is a lot of snow'],
      'neg' : ['it does not rain', 'there is not a lot of snow']}

con = {'pos': ['the trees do look beautiful', 'it is very cold'],
      'neg' : ['the trees do not look beautiful', 'it is not very cold']}

sents_A = []
sents_B = []
labels = []


for conn in connec:
    for p_i in np.arange(len(phe['pos'])):
        for c_i in np.arange(len(con['pos'])):
        
            pcon = con['pos'][c_i]
            ncon = con['neg'][c_i]
        
            pphe = phe['pos'][p_i]
            nphe = phe['neg'][p_i]
        
            sents_A.append(pcon + " " + conn + " " + pphe + ' . ')
            sents_B.append(ncon + " " + conn + " " + pphe + ' . ')
            labels.append('CONTRADICTION')
        
            sents_A.append(pcon + " " + conn + " " + pphe + ' . ')
            sents_B.append(pcon + " " + conn + " " + nphe + ' . ')
            labels.append('NEUTRAL')
        
            # self-rep/rephrase
            sents_A.append(pcon + " when " + pphe + ' . ')
            sents_B.append('When ' + pphe + ', ' + pcon + ' . ')
            labels.append('ENTAILMENT')
        
    #         # two nots
    #         sents_A.append(pcon + " when " + pphe + ' . ')
    #         sents_B.append(ncon + " when " + nphe + ' . ')
    #         labels.append('NEUTRAL')
        
    #         # Rephrase
    #         sents_A.append(pcon + " when " + pphe + ' . ')
    #         sents_B.append("When " + pphe + ' , ' + ncon + ' . ')
    #         labels.append('CONTRADICTION')
        
    #         sents_A.append(pcon + " when " + pphe + ' . ')
    #         sents_B.append("When " + nphe + ' , ' + pcon + ' . ')
    #         labels.append('NEUTRAL')
        
    

open("testData/toy/s1.ncon", 'w').write("\n".join([str(x) for x in sents_A]))
open("testData/toy/s2.ncon", 'w').write("\n".join([str(x) for x in sents_B]))
np.savetxt("testData/toy/labels.ncon", [label2id[x] for x in labels], fmt='%i')



# C. With but/however/whereas discourse markers
# 
# Could also add although?

# In[6]:

# Generate discourse marked examples
discs = ['however']
nps = ["the woman in the black shirt",
       "the fat man"]

vps = ['sit down', 'walk in']

sents_A = []
sents_B = []
labels = []

for disc in discs:
    for np1 in nps:
        for np2 in nps:
            if (np1 != np2):
                for vp in vps:
                    sents_A.append(np1 + " does " + vp + " , " + disc + " " + np2 + ' does not ' + vp + ' . ')
                    sents_B.append(np1 + " does " + vp + ' . ')
                    labels.append('ENTAILMENT')
                    
                    sents_A.append(np1 + " does " + vp + " , " + disc + " " + np2 + ' does not ' + vp + ' . ')
                    sents_B.append(np1 + " does not " + vp + ' . ')
                    labels.append('CONTRADICTION')
                    
                    sents_A.append(np1 + " does " + vp + " , " + disc + " " + np2 + ' does not ' + vp + ' . ')
                    sents_B.append(np2 + " does " + vp + ' . ')
                    labels.append('CONTRADICTION')
                    
                    sents_A.append(np1 + " does " + vp + " , " + disc + " " + np2 + ' does not ' + vp + ' . ')
                    sents_B.append(np2 + " does not " + vp + ' . ')
                    labels.append('ENTAILMENT')
                    
                    sents_A.append(np1 + " does not " + vp + " , " + disc + " " + np2 + ' does ' + vp + ' . ')
                    sents_B.append(np1 + " does " + vp + ' . ')
                    labels.append('CONTRADICTION')
                    
                    sents_A.append(np1 + " does not " + vp + " , " + disc + " " + np2 + ' does ' + vp + ' . ')
                    sents_B.append(np1 + " does not " + vp + ' . ')
                    labels.append('ENTAILMENT')
                    
                    sents_A.append(np1 + " does not " + vp + " , " + disc + " " + np2 + ' does ' + vp + ' . ')
                    sents_B.append(np2 + " does " + vp + ' . ')
                    labels.append('ENTAILMENT')
                    
                    sents_A.append(np1 + " does not " + vp + " , " + disc + " " + np2 + ' does ' + vp + ' . ')
                    sents_B.append(np2 + " does not " + vp + ' . ')
                    labels.append('CONTRADICTION')
                    
                                    



open("testData/toy/s1.subjv", 'w').write("\n".join([str(x) for x in sents_A]))
open("testData/toy/s2.subjv", 'w').write("\n".join([str(x) for x in sents_B]))
np.savetxt("testData/toy/labels.subjv", [label2id[x] for x in labels], fmt='%i')


# #### Requiring bigram compositionality as well as symmetry understanding
# 
# A. Comparatives

# In[7]:

# should I do taller and not taller? Or taller and shorter?

comps_p = {'pos': ['taller', 'more cheerful'],
          'neg': ['shorter', 'less cheerful']}

comps_o = {'pos': ['bigger'],
           'neg' : ['smaller']}

comps_t = {'pos': ['longer'], 
           'neg': ['shorter']}

nps_p = [ "the girl", "the old woman"]
nps_o = ['the brown table', 'the metal chair']
nps_t = ['the art film']

sents_A = []
sents_B = []
labels = []

nps = {'obj' : nps_o,
      'pers': nps_p,
      'time': nps_t}

comps = {'obj' : comps_o,
      'pers': comps_p,
      'time': comps_t}

for key in nps:
    for np1 in nps[key]:
        for np2 in nps[key]:
            for p, n in zip(comps[key]['pos'], comps[key]['neg']):
                if (np1 != np2):
                    
#                     # words not exactly the same - one word difference
                    
                    sents_A.append(np1 + " is " + p + " than " + np2 + ' . ')
                    sents_B.append(np2 + " is " + n + " than " + np1 + ' . ')
                    labels.append('ENTAILMENT')
                    
                    sents_A.append(np1 + " is " + n + " than " + np2 + ' . ')
                    sents_B.append(np2 + " is " + p + " than " + np1 + ' . ')
                    labels.append('ENTAILMENT')

                    sents_A.append(np1 + " is " + p + " than " + np2 + ' . ')
                    sents_B.append(np1 + " is " + n + " than " + np2 + ' . ')
                    labels.append('CONTRADICTION')
                    
                    sents_A.append(np1 + " is " + n + " than " + np2 + ' . ')
                    sents_B.append(np1 + " is " + p + " than " + np2 + ' . ')
                    labels.append('CONTRADICTION')
                    
                    sents_A.append(np1 + " is " + p + " than " + np2 + ' . ')
                    sents_B.append(np2 + " is " + p + " than " + np1 + ' . ')
                    labels.append('CONTRADICTION')
                    
                    sents_A.append(np1 + " is " + n + " than " + np2 + ' . ')
                    sents_B.append(np2 + " is " + n + " than " + np1 + ' . ')
                    labels.append('CONTRADICTION')
                    
#                     # Self - rep
                    sents_A.append(np1 + " is " + p + " than " + np2 + ' . ')
                    sents_B.append(np1 + " is " + p + " than " + np2 + ' . ')
                    labels.append('ENTAILMENT')
                    
                    sents_A.append(np1 + " is " + n + " than " + np2 + ' . ')
                    sents_B.append(np1 + " is " + n + " than " + np2 + ' . ')
                    labels.append('ENTAILMENT')




open("testData/toy/s1.comp", 'w').write("\n".join([str(x) for x in sents_A]))
open("testData/toy/s2.comp", 'w').write("\n".join([str(x) for x in sents_B]))
np.savetxt("testData/toy/labels.comp", [label2id[x] for x in labels], fmt='%i')

#FUTURE
# Might be easier to just give Mturkers the _np_ is _"more"_ _adj_ _np_ framework and ask to fill?

