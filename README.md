# Evaluating compositionality in sentence embeddings
Codebase for Dasgupta et al. 2018, https://arxiv.org/abs/1802.04302

An updated version of this work is Dasgupta, Ishita, et al. "Analyzing machine-learned representations: A natural language case study." arXiv preprint arXiv:1909.05885 (2019).
Linked here: https://arxiv.org/abs/1909.05885

Code to generate Compositional dataset based on comparisons, SNLI data analysis and scripts for augmented training is in the training-exepriments branch.

Dataset used in the paper is here: https://github.com/ishita-dg/ScrambleTests/tree/training-experiment/testData/new

Code in main branch generates a smaller but more general composotional dataset, sets up classifiers, downloads and tokenizes data.

## Instructions ##
### Getting data ###
In the Downloads folder, run:
`./get_data.bash`
Requires `7za` to unzip downloaded files, download and install from https://sourceforge.net/projects/p7zip/files/p7zip/
Path to sed tokenizer might need to be adjusted.

### Run-through with toy ###
Run: `python main.py`, with `toy = True`.
This should run through training the classifier and test code on toy data sets (provided).

Setting it False will run the true classifier and take a long time, and very high memory (~150+ GB) for InferSent embeddings.

### GPU for classifier training ###
Set `useCudaReg = True` in `main.py`

### Analysing tests ###
The logistic regression models (in ./models/) as well as their outputs on the true scramble-test results (in ./regout/) are provided.
So you can run the analysis script directly.

In `AnalyseTests.ipynb`, setting `Scram = True` runs tests for Scramble test data, `Scram = False` runs it for te SNLI test and/or dev sets.
Produces the plots (inline as well as in ./figures/), and displays high-margin BOW misclassifications (inline).

