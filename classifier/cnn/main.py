import random, pickle
import numpy as np

from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D

from classifier.cnn import models as models

from data_helpers import tokenize

from config import LABEL_MARK, DENSE_LAYER_SIZE, FILTER_SIZES, DROPOUT_VAL, NUM_EPOCHS, BATCH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VALIDATION_SPLIT, LABEL_DIC, INSY_PATH, OOV_VECTOR

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
        for word,i in my_dictionary.items():
            embedding_matrix[i] = vectors[insy.word_to_index[word]]
        self.embedding_matrix = embedding_matrix

def train(corpus_file,model_file,vectors_file):
        params_obj = Params()
        #preprocess data
        insy = pickle.load(open(params_obj.insy_path,"rb"))
        preprocessing = PreProcessing()
        preprocessing.loadData(corpus_file,model_file,params_obj.label_dic,create_dictionary=True,insy=insy)
        print(preprocessing.x_train[0])
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

def predict(text_file,model_file,vectors_file,evaluation=False):

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
