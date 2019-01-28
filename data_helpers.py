import pickle
import numpy as np

from config import MAX_SEQUENCE_LENGTH

def tokenize(texts, model_file, create_dictionnary,insy=False):

     if create_dictionnary:
          my_dictionary = {}
          my_dictionary["word_index"] = {}
          my_dictionary["index_word"] = {}
          my_dictionary["word_index"]["<PAD>"] = 0
          my_dictionary["index_word"][0] = "<PAD>"
          index = 0
     else:
          with open(model_file + ".index", 'rb') as handle:
               my_dictionary = pickle.load(handle)

     data = (np.zeros((len(texts), MAX_SEQUENCE_LENGTH))).astype('int32')
     
     i = 0
     if insy:
            insy.word_to_index["<PAD>"] = 0
            insy.word_to_index["<TARGET>"]=0
            insy.index_to_word[0] = "<PAD>"
            my_dictionary["word_index"] = insy.word_to_index
            my_dictionary["index_word"] = insy.index_to_word
            
     for line in texts:
          words = line.split()[:MAX_SEQUENCE_LENGTH]
          sentence_length = len(words)
          sentence = []
          for word in words:
            if not insy:
               if word not in my_dictionary["word_index"].keys():
                    if create_dictionnary:
                         index += 1
                         my_dictionary["word_index"][word] = index
                         my_dictionary["index_word"][index] = word
                    else:
                         my_dictionary["word_index"][word] = my_dictionary["word_index"]["<PAD>"]
               sentence.append(my_dictionary["word_index"][word])
            else:
                if word in insy.word_to_index:
                    sentence.append(my_dictionary["word_index"][word])
                else:
                    sentence.append(my_dictionary["word_index"]["OOV"])


          if sentence_length < MAX_SEQUENCE_LENGTH:
               for j in range(MAX_SEQUENCE_LENGTH - sentence_length):
                    sentence.append(my_dictionary["word_index"]["<PAD>"])
          
          data[i] = sentence
          i += 1

     if create_dictionnary:
          with open(model_file + ".index", 'wb') as handle:
               pickle.dump(my_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

     return my_dictionary, data
