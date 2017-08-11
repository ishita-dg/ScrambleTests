# ScrambleTests
Codebase to 
1. Train classifiers on pairs of InferSent embedded sentences to predict NLI labels
2. Test its performance on a new Scrambled dataset

## Instructions ##
### Getting data ###
In the Downloads folder, run:
`./get_data.bash`

### Run-through with toy ###
Run: `python main.py`, with `toy = True`.
This should run through training the classifier and test code on toy data sets (provided).

Setting it False will run the true classifier and take a long time, and very high memory for embeddings.

