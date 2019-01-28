import pickle

class IndexsystemWG:
    def __init__(self,topvoc=None,voc_cutoff = None,itw=None,wti=None):
        if topvoc:
            tv = pickle.load(open(topvoc,"rb"))
            voc = [0 for i in range(len(tv))]
            for (index,word) in tv.items():
                voc[int(index)] = word
            self.m = topvoc
            self.index_to_word = ["NULL"] + voc + ["STARTSEQ","ENDSEQ","OOV"]
            self.word_to_index = {word:index for index,word in enumerate(self.index_to_word)}
            self.vocsize = len(self.word_to_index)
        else:
            self.index_to_word = itw
            self.word_to_index=wti
            self.vocsize=len(self.index_to_word)

    def get_index(self,word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return self.word_to_index["OOV"]

    def write(self,filename):
        pickle.dump(self,open(filename,"wb"))
