#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 16 nov. 2017
@author: laurent.vanni@unice.fr
'''
import sys
import os
import json
import numpy as np

from classifier.cnn.main import train, predict, get_maximal_stimuli, get_activations
from config import FILTER_SIZES

def print_help():
    print("usage: python hyperdeep.py <command> <args>\n")
    print("The commands supported by deeperbase are:\n")
    print("\ttrain\ttrain a CNN model for sentence classification\n")
    print("\tpredict\tpredict most likely labels")
    print("\tstimuli\tretrieves maximal stimuli in corpus for filters of specified width\n")
    print("\tactivation\tretrieves maximal activation score for every filter across input sentences\n")
    
def print_invalidArgs_mess():
    print("Invalid argument detected!\n")

def get_args():
    args = {}
    for i in range(2, len(sys.argv[1:])+1):
        if sys.argv[i][0] == "-":
            args[sys.argv[i]] = sys.argv[i+1]
        else:
            args[i] = sys.argv[i]
    return args

if __name__ == '__main__':

    # GET COMMAND
    try:
        command = sys.argv[1]
        if command not in ["train", "predict","stimuli","activation"]:
            raise
    except:
        print_help()
        exit()

    # EXECT COMMAND
    if command == "train":
        try:
            args = get_args()
            corpus_file = args["-input"]
            model_file = args["-output"]
            train(corpus_file, model_file, args.get("-w2vec", False))
        except:
            raise
            print_invalidArgs_mess()
            print("The following arguments are mandatory:\n")
            print("\t-input\ttraining file path")
            print("\t-output\toutput file path\n")
            print("The following arguments for training are optional:\n")
            print("\t-w2vec\tword vector representations file path\n")
            print_help()
            exit()

    if command == "predict":
        #try:
            args = get_args()
            model_file = args[2]
            vectors_file = args[3]
            text_file = args[4]
            output = args[5]
            compressed = args[6]
            average = args[7]
            if compressed == "False":
                compressed = False
            if average == "False":
                average = False
            # save predictions in a file
            if not compressed:
                predictions = predict(text_file,model_file,vectors_file,compressed,average)
                result_path = output
                results = open(result_path, "w")
                results.write(json.dumps(predictions,indent=None,separators=(",\n",": ")))
                results.close()
            else:
                TDSs,CONF = predict(text_file,model_file,vectors_file,compressed,average)
                result_path = output
                if not average:
                    for fs,TDS in zip(FILTER_SIZES,TDSs):
                        np.save(result_path +"." + str(fs) + ".TDS",TDS)
                np.save(result_path + ".CONF",CONF)
    if command == "stimuli":
            args = get_args()
            model_file = args[2]
            vectors_file = args[3]
            text_file = args[4]
            output = args[5]
            filtersize = int(args[6])
            max_rank = int(args[7])
            filters = get_maximal_stimuli(text_file,model_file,vectors_file,filtersize,max_rank)
            opf = open(output,"w")
            opf.write("FEATURE_NR\tRANK_OF_STIMULUS\tSTIMULUS\tACTIVATION_SCORE\n")
            for featurenr,filter in enumerate(filters):
                for (stimulusnr,feature) in enumerate(filter):
                    (stimuli,score) = feature
                    opf.write("%s\t%s\t%s\t%s\n" % (featurenr,stimulusnr,stimuli,score))
            opf.close()
    if command == "activation":
            args = get_args()
            model_file = args[2]
            vectors_file = args[3]
            text_file = args[4]
            output = args[5]
            averages = get_activations(text_file,model_file,vectors_file)
            opf = open(output,"w")
            opf.write("WIDTH_ORDER\tFEATURENR\tAVERAGE\n")
            N_FILTERSIZES=3
            nb_filters_per_size = len(averages)/N_FILTERSIZES
            for i,avg in enumerate(averages):
                width = int(i//nb_filters_per_size)
                featurenr = int(i%nb_filters_per_size)
                opf.write("%s\t%s\t%s\n" % (width,featurenr,avg))
            opf.close()


        #except:
        #    print_invalidArgs_mess()
        #    print("usage: hyperdeep predict <model> <vec> <test-data>:\n")
        #    print("\t<model>\tmodel file path\n")
        #    print("\t<vec>\tword vector representations file path\n")
        #    print("\t<test-data>\ttest data file path\n")
        #    print_help()
        #    exit()

