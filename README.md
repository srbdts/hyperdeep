
# Overview
- hyperdeep.py: main script you call to train a model, predict input data, get the filter stimuli or average activation scores
- config.py: contains the hyperparameters of the model. "label_dic" needs to contain the output classes in your dataset.
- data_helpers.py: tokeniseert de inputdata; wordt aangeroepen vanuit classifier/cnn/main.py
- classifier/cnn/models.py: creates the model; called from classifier/cnn/main.py
- classifier/cnn/main.py: prepares the data, makes the call to create the model, trains the model and saves it.


# Train a model
python hyperdeep.py train -input <INPUTFILE> -output <OUTPUTMODEL> -w2vec <EMBEDDINGS>
e.g. python hyperdeep.py train -input data/p1_100K.data -output results/test.out -w2vec embeddings/em_p1_bi.npy


# Have a trained model label input sentences
python hyperdeep.py predict <MODEL> <EMBEDDINGS> <INPUTFILE> <OUTPUTFILE> <COMPRESSED>
e.g. python hyperdeep.py predict results/test.out embeddings/em_p1_bi.npy data/p1_1K.data results/p1_1K.json False

<COMPRESSED> is a boolean that allows you to specify the output format. If set to False, the output will be a json-file that contains the input sentences along with TDS-scores for each word and prediction scores for each sentence. If set to True, the input sentences themselves will not be outputted again. Instead, you will receive one numpy matrix with the prediction scores for each sentence, and one numpy matrix with TDS scores for every filter size. For example, assume that the inputfile contains 100 sentences of 50 words; the model contains 100 filters of size 3, 100 of size 4 and 100 of size 5; and the model has been trained to distinguish 3 classes. In that case, the output matrix with the prediction scores will be 100x3 (#sentences x #classes), and the 3 (#filtersizes) TDS matrices will be 100x50 (#sentences x #words).


# For every filter, find the N-grams in the input data that activate that filter most
python hyperdeep.py stimuli <MODEL> <EMBEDDINGS> <INPUTFILE> <OUTPUTFILE> <FILTERSIZE> <MAX_RANK>
e.g. python hyperdeep.py stimuli results/test.out embeddings/em_p1_bi.npy data/p1_1K.data results/p1_stimuli.txt 3 50

<FILTERSIZE> and <MAX_RANK> need to be integers. <FILTERSIZE> specifies the width of the filters that you want to fetch the stimuli for. <MAX_RANK> specifies how many stimuli you want for each filter (e.g. "10" if you want to retrieve the 10 strongest stimuli for every filter)

Useful if you want to inspect what each filter has grown sensitive to.


# Get average activation score for each filter in the input sentences you provide
python hyperdeep.py activation <MODEL> <EMBEDDINGS> <INPUTFILE> <OUTPUTFILE>
e.g. python hyperdeep.py activation results/test.out embeddings/em_p1_bi.npy data/p1_1K.data results/p1_activations.txt 3

If the input file contains only input sentences of the same class, the activations indicate to what extent each filter is activated by that class. Paired with the output of 'stimuli', this allows you to determine which pattern every filter has grown sensitive to, and to what extent that pattern is attested in the data of that particular class.
