import pickle
import numpy as np

from config import MAX_SEQUENCE_LENGTH

def tokenize(texts, model_file, create_dictionnary,insy=False):
    
     # initialize variables

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

     # If indexsystem has been provided, make sure that it contains words for padding and target
     if insy:
            insy.word_to_index["<PAD>"] = 0
            insy.word_to_index["<TARGET>"]=len(insy.word_to_index)
            insy.index_to_word[0] = "<PAD>"
            insy.index_to_word.append("<TARGET>")
            my_dictionary["word_index"] = insy.word_to_index
            my_dictionary["index_word"] = insy.index_to_word
     
     # Loop through input sentences and vectorze them       
     for line in texts:
          words = line.split()[:MAX_SEQUENCE_LENGTH]
          sentence_length = len(words)
          sentence = []
          for word in words:
               # If no indexsystem was provided, add new words to vocabulary as you encounter them
               if not insy:
                    if word not in my_dictionary["word_index"].keys():
                         if create_dictionnary:
                              index += 1
                              my_dictionary["word_index"][word] = index
                              my_dictionary["index_word"][index] = word
                         else:
                              my_dictionary["word_index"][word] = my_dictionary["word_index"]["<PAD>"]
                    sentence.append(my_dictionary["word_index"][word])
               # If indexsystem was provided, use that index mapping for vectorization 
               else:
                    if word in insy.word_to_index:
                         sentence.append(my_dictionary["word_index"][word])
                    else:
                         sentence.append(my_dictionary["word_index"]["OOV"])
    
          # If sentence is shorter than other sentences in its batch, add zero-padding
          if sentence_length < MAX_SEQUENCE_LENGTH:
               for j in range(MAX_SEQUENCE_LENGTH - sentence_length):
                    sentence.append(my_dictionary["word_index"]["<PAD>"])
          
          data[i] = sentence
          i += 1
     
     # Save indexation along with model
     if create_dictionnary:
          with open(model_file + ".index", 'wb') as handle:
               pickle.dump(my_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

     return my_dictionary, data
