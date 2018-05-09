from gensim.models import KeyedVectors
from random_vec import RandomVec

class fastText():
    def __init__(self, filename, dim=100):
        self.model = {}
        self.rand_model = RandomVec(dim)
        try:
            self.wvec_model = KeyedVectors.load_word2vec_format(filename)
        except:
            print("Please provide fastText embedding model [filename].vec")

    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.wvec_model[word]
        except:
            return self.rand_model[word]
