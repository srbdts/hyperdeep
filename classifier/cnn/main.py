import random, pickle
import numpy as np
from numpy import unravel_index

from keras.utils import np_utils
from keras.models import load_model,Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Conv2DTranspose

from classifier.cnn import models as models

from data_helpers import tokenize

from config import LABEL_MARK, DENSE_LAYER_SIZE, FILTER_SIZES, NB_FILTERS, DROPOUT_VAL, NUM_EPOCHS, BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VALIDATION_SPLIT, LABEL_DIC, INSY_PATH, OOV_VECTOR, MASKER

from indexsystemwg import *

class Params:
    
    # initialize default parameters
    dense_layer_size = DENSE_LAYER_SIZE
    filter_sizes = FILTER_SIZES
    dropout_val = DROPOUT_VAL
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    inp_length = MAX_SEQUENCE_LENGTH
    embeddings_dim = EMBEDDING_DIM
    label_dic = LABEL_DIC
    insy_path = INSY_PATH

class PreProcessing:

    def loadData(self, corpus_file, model_file, label_dic, create_dictionary,insy,preserve_order=False,evaluate=True):
        """Loads input data into memory; splits them into input, output, training and validation data """
        
        # Initialize variables
        print("loading data...")
        self.corpus_file = corpus_file

        labels = []
        texts = []

        # Read text and detect classes/labels
        self.num_classes = len(label_dic)
        f = open(corpus_file,"r")
        for text in f.readlines():
            label = text.split(LABEL_MARK + " ")[0].replace(LABEL_MARK,"")
            text = text.replace(LABEL_MARK + label + LABEL_MARK + " ","")
            if evaluate:
                label_int = label_dic[label]
            else:
                label_int = 0
            labels += [label_int]
            texts += [text]
        f.close()
        
        # Tokenize and vectorize input texts by means of method in "data_helpers.py"
        my_dictionary, data = tokenize(texts, model_file, create_dictionary,insy)

        print("Found %s unique tokens." % len(my_dictionary["word_index"]))

        labels = np_utils.to_categorical(np.asarray(labels))

        print("Labels: %s" % (" ".join([str(label) + "/" + str(int) for label,int in label_dic.items()])))
        
        # If training a model, randomize the order of the input sentences. If predicting, preserve original order
        if not preserve_order:
           print("shuffling indices")
           indices = np.arange(data.shape[0])
           np.random.shuffle(indices)
           data = data[indices]
           labels = labels[indices]
        
        # Split into training and testset
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        self.x_train = data[:-nb_validation_samples]
        self.y_train = labels[:-nb_validation_samples]
        self.x_val = data[-nb_validation_samples:]
        self.y_val = labels[-nb_validation_samples:]
        self.my_dictionary = my_dictionary

    def loadEmbeddingsCustom(self,vectors_file,insy):
        """Custom method to load input embeddings compatible with the embeddings I created with gensim and my custom indexsystem"""
        
        # Load embedding vectors
        vectors = np.load(vectors_file)
        
        # Load indexsystem
        words = insy.index_to_word
        my_dictionary = self.my_dictionary["word_index"]
       
        # Initialize final embedding matrix
        embedding_matrix = np.zeros((len(my_dictionary)+1,EMBEDDING_DIM))
        
        # Get random vector for out of vocabulary words and target words
        oov_vector = np.random.rand(EMBEDDING_DIM)
        target_vector = np.random.rand(EMBEDDING_DIM)
        # Iterate over words found in input texts and map them to the right embedding vectors
        for word,i in my_dictionary.items():
            if  word == "<TARGET>":
                if MASKER == "target":
                    embedding_matrix[i] = target_vector
                elif MASKER == "oov":
                    embedding_matrix[i] = oov_vector
                elif MASKER == "zero":
                    continue
            elif word == "OOV":
                embedding_matrix[i] = oov_vector
            else:
                embedding_matrix[i] = vectors[insy.word_to_index[word]] 
        self.embedding_matrix = embedding_matrix

def train(corpus_file,model_file,vectors_file):
        """Train convolutional network"""
        
        # Load input data into memory and vectorize
        params_obj = Params()
        insy = pickle.load(open(params_obj.insy_path,"rb"))
        preprocessing = PreProcessing()
        preprocessing.loadData(corpus_file,model_file,params_obj.label_dic,create_dictionary=True,insy=insy)
        preprocessing.loadEmbeddingsCustom(vectors_file,insy)

        # Set additional parameters
        params_obj.num_classes = preprocessing.num_classes
        params_obj.vocab_size = len(preprocessing.my_dictionary["word_index"])
        params_obj.inp_length = MAX_SEQUENCE_LENGTH
        params_obj.embeddings_dim = EMBEDDING_DIM

        # Define and compile model
        cnn_model = models.CNNModel()
        model = cnn_model.getModel(params_obj=params_obj,weight=preprocessing.embedding_matrix)
        
        # Train model
        x_train,y_train,x_val,y_val = preprocessing.x_train, preprocessing.y_train,preprocessing.x_val, preprocessing.y_val
        checkpoint = ModelCheckpoint(model_file,monitor="val_acc",verbose=1,save_best_only=True,mode="max")
        callbacks_list = [checkpoint]
        model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=params_obj.num_epochs,batch_size=params_obj.batch_size,callbacks=callbacks_list)


def get_maximal_stimuli(text_file,model_file,vectors_file,filtersize,max_rank):
        """For every filter of the specified size in the pretrained model, retrieve from the input sentences the n (max_rank) sentence fragments that activate that filter most."""
        
        # Load input data and pretrained model into memory; vectorize input
        params_obj = Params()
        insy = pickle.load(open(params_obj.insy_path,"rb"))
        preprocessing = PreProcessing()
        preprocessing.loadData(text_file,model_file,params_obj.label_dic,create_dictionary=False,insy=insy,preserve_order=True,evaluate=False)
        preprocessing.loadEmbeddingsCustom(vectors_file,insy)
        preprocessing.my_dictionary["index_word"].append("<TARGET>")
        x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val),axis=0)
        full_model = load_model(model_file)
        
        # Copy original model until the convolution layer with the filters of the specified width and store as new model
        target_layer = "conv2d_" + str(FILTER_SIZES.index(filtersize)+1)
        model = Model(inputs=full_model.get_layer("input_1").input,outputs=full_model.get_layer(target_layer).output)
        
        # Traverse the input sentences and find the location of the highest activation functions for every filter
        results = []
        predictions = model.predict(x_data)
        # For every filter of the specified width
        for i in range(predictions.shape[-1]):
                # Compute activation matrix of all input sentences for that filter alone
                slice = predictions[:,:,:,i]
                if not len(results) > i:
                    results.append([])
                for rank in range(max_rank):
                    # get position (sentence number and position of word within sentence) of maximal stimuli
                    (n_sentence,n_word,_) = unravel_index(np.argmax(slice),slice.shape)
                    # get activation value caused by maximal stimuli
                    max = np.max(slice)
                    stimuli = []
                    # collect the words at the corresponding locations
                    for sliding_window in range(filtersize):
                        stimuli.append(preprocessing.my_dictionary["index_word"][x_data[n_sentence][n_word+sliding_window]])
                        if n_word+sliding_window == len(x_data[n_sentence])-1:
                            break
                    results[i].append((" ".join(stimuli),max))
                    # Set the activation value to 0 so that the following iteration finds the next maximal value as maximum.
                    slice[(n_sentence,n_word,0)]=0
        
        #sorted_results = []
        #for filterresults in results:
        #    filterresults.sort(key=lambda x: x[1])
        #    sorted_results.append([stimulus for (stimulus,score) in filterresults[:max_rank]])        
        
        return results

def get_activations(text_file,model_file,vectors_file):
    """Compute to what extent every filter was activated on average in the provided set of input sentences"""

    # Load input data and pretrained model into memory; vectorize input
    params_obj = Params()
    insy = pickle.load(open(params_obj.insy_path,"rb"))
    preprocessing = PreProcessing()
    preprocessing.loadData(text_file,model_file,params_obj.label_dic,create_dictionary=False,insy=insy,preserve_order=True,evaluate=False)
    preprocessing.loadEmbeddingsCustom(vectors_file,insy)
    index_to_word = preprocessing.my_dictionary["index_word"]
    x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val),axis=0)
    full_model = load_model(model_file)
    
    # Copy the original model until the concatenate layer and store as new model
    model = Model(inputs=full_model.get_layer("input_1").input,outputs=full_model.get_layer("concatenate_1").output)
    
    # Predict concatenated activation bars all input sentences
    predictions = model.predict(x_data)
    
    # Compute average activation for each value in the concatenated activation bars throughout the entire input set
    averages = []
    for filterwidth_number in range(predictions.shape[1]):
        for filter_number in range(predictions.shape[-1]):
            slice = predictions[:,filterwidth_number,0,filter_number]
            averages.append(np.average(slice))
    return averages

def predict(text_file,model_file,vectors_file,compressed=False):
        """Predict output labels for input sentences based on pretrained model file."""
        
        # Load input data and pretrained model into memory; vectorize input
        result = []
        params_obj = Params()
        insy = pickle.load(open(params_obj.insy_path,"rb"))
        preprocessing = PreProcessing()
        preprocessing.loadData(text_file,model_file,params_obj.label_dic,create_dictionary=False,insy=insy,preserve_order=True,evaluate=False)
        preprocessing.loadEmbeddingsCustom(vectors_file,insy)
        x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val),axis=0)
        model = load_model(model_file)
    
        # Predict output class for all input sentences
        predictions = model.predict(x_data)
        
        # Build deconvolution models:
        filternr = 1
        deconv_models = []
        for filterwidth in FILTER_SIZES:
            # Take confolution layer of specified width
            inputlayer = "conv2d_" + str(filternr)
            # Create new deconvolution layer of same width and attach it to output of convolution layer 
            deconv_layer_bis = Conv2DTranspose(1,(filterwidth,EMBEDDING_DIM),padding="valid",data_format="channels_last",kernel_initializer="normal",activation="relu")(model.get_layer(inputlayer).output)
            # Create new model with same layers as original model until the specifiec convolutional layers, but with extra deconvolution layer
            deconv_model = Model(inputs=model.input,outputs=deconv_layer_bis)
            # Set the weights of the deconvolutional layer to equal the convolutional weights learned during training
            for layer in deconv_model.layers:
                if type(layer) is Conv2D:
                    deconv_weights = layer.get_weights()[0]
            deconv_bias = deconv_model.layers[-1].get_weights()[1]
            deconv_model.layers[-1].set_weights([deconv_weights,deconv_bias])
            deconv_models.append(deconv_model)
            filternr+= 1 
        

        # Predict deconvolution values ( = TDS scores) for every word in every input sentence
        deconvs = []
        for deconv_model in deconv_models:
            deconvs.append(deconv_model.predict(x_data))

        # Write output
        my_dictionary = preprocessing.my_dictionary
        if compressed:
            TDSs=[]
            for deconv in deconvs:
                #Sum over 1-dimensional axis
                deconv = np.sum(deconv,axis=-1)
                #Sum over 400-dimensional axis
                deconv = np.sum(deconv,axis=-1)
                TDSs.append(np.asarray(deconv,dtype="int16"))
            CONF = np.asarray(predictions,dtype="float32")
            return TDSs,CONF
        else:
            TDSs = []
            for deconv in deconvs:
                deconv = np.sum(deconv,axis=-1)
                deconv = np.sum(deconv,axis=-1)
                TDSs.append(np.asarray(deconv,dtype="int16"))
            for sentence_nb in range(len(x_data)):
                sentence = {}
                sentence["sentence"] = ""
                sentence["prediction"] = predictions[sentence_nb].tolist()
                for i in range(len(x_data[sentence_nb])):
                    word = ""
                    index = x_data[sentence_nb][i]
                    try:
                        word = my_dictionary["index_word"][index]
                    except:
                        word = "PAD"
                    # READ DECONVOLUTION
                    deconv_values = []
                    for TDS in TDSs:
                        deconv_values.append(str(TDS[sentence_nb][i]))
                    sentence["sentence"] += word + ":" + "/".join(deconv_values) + " "
                result.append(sentence)
            return result
        
