import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim import utils
import tempfile

class LyricEmbedding:
    def __init__(self, lyric_path = "TravisScott", save = True, embedding_filename = "travis_embedding"):
        self.model = None
        self.lyric_path = lyric_path
        self.embedding_filename = embedding_filename
        if os.path.exists(self.embedding_filename):
            self.__load_model()
        else:
            corpus = self.__load_lyrics()
            self.print_statistics(corpus)
            self.__make_model(corpus)
            if save: self.__save_model()


    def __load_lyrics(self):
        """
        Load the dataset
        """
        if os.path.exists(self.lyric_path):
            corpus = []
            for file_name in os.listdir(self.lyric_path):
                file_path = os.path.join(self.lyric_path, file_name)
                if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == ".txt":
                    for line in open(file_path):
                        corpus.append(utils.simple_preprocess(line))
            return corpus

        else:
            raise Exception("Lyric path does not exist")

    def __make_model(self, corpus):
        """
        Embed data, Word2Vec
        """
        # TODO - Tune Parameters
        self.model = Word2Vec(min_count=1, window=2, sample=0.001, alpha=0.03, min_alpha=0.0007, negative=20, workers=3)
        self.model.build_vocab(corpus, progress_per=10)

    def __save_model(self):
        if self.model:
            with tempfile.NamedTemporaryFile() as tmp:
                self.model.save(self.embedding_filename)
        else:
            raise Exception("Model cannot be saved as it does not exist")
    
    def __load_model(self):
        if os.path.exists(self.embedding_filename):
            with tempfile.NamedTemporaryFile() as tmp:
                self.model = Word2Vec.load(self.embedding_filename)
        else:
            raise Exception("Model cannot be loaded as it does not exist")

    def print_statistics(self, corpus):
        """
        Print statistics of dataset to ensure data quality
        """
        print("Number of lines: {}".format(len(corpus)))

        total_length = 0
        for line in corpus:
            total_length += len(line)

        print("Average lyric length: {}".format(total_length/len(corpus)))

    def get_wv(self):
        """
        Returns the word vector of the model
        """
        return self.model.wv
