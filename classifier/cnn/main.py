import random, pickle
import numpy as np
from numpy import unravel_index

from keras.utils import np_utils
from keras.models import load_model,Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D

from classifier.cnn import models as models

from data_helpers import tokenize

from config import LABEL_MARK, DENSE_LAYER_SIZE, FILTER_SIZES, DROPOUT_VAL, NUM_EPOCHS, BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VALIDATION_SPLIT, LABEL_DIC, INSY_PATH, OOV_VECTOR, MASKER

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

        my_dictionary, data = tokenize(texts, model_file, create_dictionary,insy)

        print("Found %s unique tokens." % len(my_dictionary["word_index"]))

        labels = np_utils.to_categorical(np.asarray(labels))

        print("Labels: %s" % (" ".join([str(label) + "/" + str(int) for label,int in label_dic.items()])))

        if not preserve_order:
           print("shuffling indices")
           indices = np.arange(data.shape[0])
           np.random.shuffle(indices)
           data = data[indices]
           labels = labels[indices]
        
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        self.x_train = data[:-nb_validation_samples]
        self.y_train = labels[:-nb_validation_samples]
        self.x_val = data[-nb_validation_samples:]
        self.y_val = labels[-nb_validation_samples:]
        self.my_dictionary = my_dictionary

    def loadEmbeddingsCustom(self,vectors_file,insy):
        vectors = np.load(vectors_file)
        words = insy.index_to_word
        my_dictionary = self.my_dictionary["word_index"]
        embedding_matrix = np.zeros((len(my_dictionary)+1,EMBEDDING_DIM))
        oov_vector = np.random.rand(EMBEDDING_DIM)
        target_vector = np.random.rand(EMBEDDING_DIM)
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
        params_obj = Params()
        #preprocess data
        insy = pickle.load(open(params_obj.insy_path,"rb"))
        preprocessing = PreProcessing()
        preprocessing.loadData(corpus_file,model_file,params_obj.label_dic,create_dictionary=True,insy=insy)
        preprocessing.loadEmbeddingsCustom(vectors_file,insy)

        #establish params
        params_obj.num_classes = preprocessing.num_classes
        params_obj.vocab_size = len(preprocessing.my_dictionary["word_index"])
        params_obj.inp_length = MAX_SEQUENCE_LENGTH
        params_obj.embeddings_dim = EMBEDDING_DIM

        #create and get model
        cnn_model = models.CNNModel()
        model, deconv_model = cnn_model.getModel(params_obj=params_obj,weight=preprocessing.embedding_matrix)
        #train model
        x_train,y_train,x_val,y_val = preprocessing.x_train, preprocessing.y_train,preprocessing.x_val, preprocessing.y_val
        checkpoint = ModelCheckpoint(model_file,monitor="val_acc",verbose=1,save_best_only=True,mode="max")
        callbacks_list = [checkpoint]
        model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=params_obj.num_epochs,batch_size=params_obj.batch_size,callbacks=callbacks_list)

        #save deconv model
        i=0
        for layer in model.layers:
            weights = layer.get_weights()
            deconv_model.layers[i].set_weights(weights)
            i+= 1
            if type(layer) is Conv2D:
                break
        deconv_model.save(model_file+".deconv")

def get_maximal_stimuli(text_file,model_file,vectors_file,filtersize,max_rank):
        
        params_obj = Params()
        insy = pickle.load(open(params_obj.insy_path,"rb"))
        preprocessing = PreProcessing()
        preprocessing.loadData(text_file,model_file,params_obj.label_dic,create_dictionary=False,insy=insy,preserve_order=True,evaluate=False)
        preprocessing.loadEmbeddingsCustom(vectors_file,insy)
        index_to_word = preprocessing.my_dictionary["index_word"]
        index_to_word.append("<TARGET>")
        x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val),axis=0)
        full_model = load_model(model_file)
        target_layer = "conv2d_" + str(filtersize-4) 
        model = Model(inputs=full_model.get_layer("input_1").input,outputs=full_model.get_layer(target_layer).output)
        print(x_data.shape)
        #nr_batches = x_data.shape[0]//100000 +1
        results = []
        #for batch in range(nr_batches):
        #    predictions = model.predict(x_data[100000*batch:min(len(x_data),100000*(batch+1))])
        predictions = model.predict(x_data)
        for i in range(predictions.shape[-1]):
        #for i in range(5):
                slice = predictions[:,:,:,i]
                if not len(results) > i:
                    results.append([])
                for rank in range(max_rank):
                    (n_sentence,n_word,_) = unravel_index(np.argmax(slice),slice.shape)
                    max = np.max(slice)
                    stimuli = []
                    for sliding_window in range(filtersize):
                        stimuli.append(preprocessing.my_dictionary["index_word"][x_data[n_sentence][n_word+sliding_window]])
                        if n_word+sliding_window == len(x_data[n_sentence])-1:
                            break
                #print(np.max(slice))
                    results[i].append((" ".join(stimuli),max))
                    slice[(n_sentence,n_word,0)]=0
        sorted_results = []
        for filterresults in results:
            filterresults.sort(key=lambda x: x[1])
            sorted_results.append([stimulus for (stimulus,score) in filterresults[:max_rank]])        
        print(len(sorted_results))
        return sorted_results

def get_activations(text_file,model_file,vectors_file):
    params_obj = Params()
    insy = pickle.load(open(params_obj.insy_path,"rb"))
    preprocessing = PreProcessing()
    preprocessing.loadData(text_file,model_file,params_obj.label_dic,create_dictionary=False,insy=insy,preserve_order=True,evaluate=False)
    preprocessing.loadEmbeddingsCustom(vectors_file,insy)
    index_to_word = preprocessing.my_dictionary["index_word"]
    index_to_word.append("<TARGET>")
    x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val),axis=0)
    full_model = load_model(model_file)
    model = Model(inputs=full_model.get_layer("input_1").input,outputs=full_model.get_layer("concatenate_1").output)
    predictions = model.predict(x_data)
    averages = []
    for filterwidth in range(3):
        for feature in range(50):
            slice = predictions[:,filterwidth,0,feature]
            averages.append(np.average(slice))
    return averages

def predict(text_file,model_file,vectors_file,evaluation=False,compressed=False):

        result = []
        params_obj = Params()
        insy = pickle.load(open(params_obj.insy_path,"rb"))
        preprocessing = PreProcessing()
        preprocessing.loadData(text_file,model_file,params_obj.label_dic,create_dictionary=False,insy=insy,preserve_order=True,evaluate=False)
        preprocessing.loadEmbeddingsCustom(vectors_file,insy)

        x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val),axis=0)
        model = load_model(model_file)
        if evaluation:
            target_word = text_file.split("/")[-1][2:-2]
            target_index = LABEL_DIC[target_word]
            targets = np.zeros((len(x_data),len(LABEL_DIC)))
            for i in range(len(targets)):
                targets[i][target_index] = 1
            evaluations = model.evaluate(x_data,targets)
            print("test loss: ", evaluations[0])
            print("test accuracy: ", evaluations[1])
    
        predictions = model.predict(x_data)
        print("------------------------------")
        print("DECONVOLUTION")
        print("------------------------------")

        deconv_model = load_model(model_file+".deconv")

        for layer in deconv_model.layers:
            if type(layer) is Conv2D:
                deconv_weights = layer.get_weights()[0]
        deconv_bias = deconv_model.layers[-1].get_weights()[1]
        deconv_model.layers[-1].set_weights([deconv_weights,deconv_bias])

        deconv = deconv_model.predict(x_data)

        my_dictionary = preprocessing.my_dictionary
        
        if compressed:
            #Sum over 1-dimensional axis
            deconv = np.sum(deconv,axis=-1)
            #Sum over 400-dimensional axis
            deconv = np.sum(deconv,axis=-1)
            #print("Shape of CONF: %s %s" % (predictions.shape()))
            #print("Shape of TDS: %s %s" % (deconv.shape()))
            TDS = np.asarray(deconv,dtype="int16")
            CONF = np.asarray(predictions,dtype="float32")
            print("--------------------------------")
            return TDS,CONF
        else:
            for sentence_nb in range(len(x_data)):
                sentence = {}
                sentence["sentence"] = ""
                sentence["prediction"] = predictions[sentence_nb].tolist()
                #if not targets:
                sentence["author"] = "UNKNOWN"
                #else:
                #    sentence["author"] = target[sentence_nb].tolist()
                for i in range(len(x_data[sentence_nb])):
                    word = ""
                    index = x_data[sentence_nb][i]
                    try:
                        word = my_dictionary["index_word"][index]
                    except:
                        word = "PAD"
                    # READ DECONVOLUTION
                    deconv_value = float(np.sum(deconv[sentence_nb][i]))
                    sentence["sentence"] += word + ":" + str(deconv_value) + " "
                    result.append(sentence)
            print("---------------------------------")
            return result
        
