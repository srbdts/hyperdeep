#!/usr/bin/python
# -*- coding: utf-8 -*-

import gensim
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.models import Model
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.text import Tokenizer

from config import embedding_dim
import numpy as np

def create_vectors(corpus_file, vectors_file=False):
    
    corpus = open(corpus_file).readlines()
    
    corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    V = len(tokenizer.word_index) + 1
    print("vocabulary_size: ", V)
    
    w_inputs = Input(shape=(1, ), dtype='int32')
    w = Embedding(V, embedding_dim)(w_inputs)
    
    # context
    c_inputs = Input(shape=(1, ), dtype='int32')
    c  = Embedding(V, embedding_dim)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)
    
    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.summary()
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')
    
    for _ in range(5):
        loss = 0.
        for i, doc in enumerate(tokenizer.texts_to_sequences(corpus)):
            data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=5, negative_samples=5.)
            x = [np.array(x) for x in zip(*data)]
            y = np.array(labels, dtype=np.int32)
            if x:
                loss += SkipGram.train_on_batch(x, y)
    
        print(loss)

    vectors = SkipGram.get_weights()[0]    
    w2v = []
    if vectors_file:
        f = open(vectors_file ,'w')
        f.write('{} {}\n'.format(V-1, embedding_dim))
        for word, i in tokenizer.word_index.items():
            vector = '{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :]))))
            f.write(vector)
            w2v += [vector]
        f.close()
    else:
        for word, i in tokenizer.word_index.items():
            w2v += ['{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :]))))]

    return w2v
    
"""
GET W2VEC MODEL
"""
def get_w2v(vectors_file):
    return gensim.models.KeyedVectors.load_word2vec_format(vectors_file, binary=False)

"""
GET WORD VECTOR
"""
def get_vector(word, w2v):
    pass  
 
"""
FIND MOST SIMILAR WORD
"""
def get_most_similar(word, vectors_file):
    w2v = get_w2v(vectors_file)
    return w2v.most_similar(positive=[word])

