# ScrambleTests
Codebase to 
1. Train classifiers on pairs of InferSent embedded sentences to predict NLI labels
2. Test its performance on a new Scrambled dataset

## Instructions ##
Downloads:
1. SNLI database to Downloads/SNLI/true

https://nlp.stanford.edu/projects/snli/snli_1.0.zip

toy SNLI provided, download the rest

`DATA_PATH = './Downloads/SNLI/toy/'`

OR

`DATA_PATH = './Downloads/SNLI/true/'`

2. glove file to Downloads/

http://nlp.stanford.edu/data/glove.840B.300d.zip

`GLOVE_PATH = './Downloads/glove.840B.300d.txt'`

3. Infersent pickled model to Downloads/

`curl -Lo encoder/infersent.allnli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle`

`MODEL_PATH = './Downloads/infersent.allnli.pickle'`

creating folders (From the home file of repo):
1. ./models/
2. ./regout/



Keeping "toy = True" in main.py should run through training the classifier and test code on toy data sets (provided).
Setting it False will run the true classifier and take a long time, and very high memory for embeddings.

