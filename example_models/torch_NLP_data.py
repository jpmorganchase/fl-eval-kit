import torch
from torchnlp.datasets import imdb_dataset
from torch.utils.data import TensorDataset, DataLoader
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
import numpy as np
import pandas as pd

class LoadData:

    def __init__(self) -> None:
        pass

    def load_data(self):

        # Load the imdb training dataset
        train = imdb_dataset(train=True)
        test = imdb_dataset(test=True)
        # RETURNS: {'text': 'For a movie that gets..', 'sentiment': 'pos'}

        return train, test

    @staticmethod
    def preprocess_string(s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with no space
        s = re.sub(r"\s+", '', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)

        return s
    
    @staticmethod
    def tockenize(x_train,y_train,x_val,y_val):
        word_list = []

        stop_words = set(stopwords.words('english')) 
        for sent in x_train:
            for word in sent.lower().split():
                word = LoadData.preprocess_string(word)
                if word not in stop_words and word != '':
                    word_list.append(word)
    
        corpus = Counter(word_list)
        # sorting on the basis of most common words
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:200]
        # creating a dict
        onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
        
        # tockenize
        final_list_train,final_list_test = [],[]
        for sent in x_train:
                final_list_train.append([onehot_dict[LoadData.preprocess_string(word)] for word in sent.lower().split() 
                                        if LoadData.preprocess_string(word) in onehot_dict.keys()])
        for sent in x_val:
                final_list_test.append([onehot_dict[LoadData.preprocess_string(word)] for word in sent.lower().split() 
                                        if LoadData.preprocess_string(word) in onehot_dict.keys()])
                
        encoded_train = [1 if label =='pos' else 0 for label in y_train]  
        encoded_test = [1 if label =='pos' else 0 for label in y_val] 

        return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test), onehot_dict

    @staticmethod
    def padding_(sentences, seq_len):
        features = np.zeros((len(sentences), seq_len),dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features
    

    def preprocess_and_return_tokens(self, data=None, v_padding=500, batch_size=50, shuffle=True):

        if data is None:
             data_train_list, data_test_list = self.load_data()
             data_train=pd.DataFrame(data_train_list[0:100])
             data_test=pd.DataFrame(data_test_list[0:100])
        else:
             data_train = data[0][0:100]
             data_test = data[1][0:100]

        x_train, y_train, x_test, y_test, vocab = LoadData.tockenize(data_train.text.values, data_train.sentiment.values, 
                                                                data_test.text.values, data_test.sentiment.values)
        
        #we have very less number of reviews with length > 500.
        #So we will consideronly those below it.
        x_train_pad = self.padding_(x_train, v_padding)
        x_test_pad = self.padding_(x_test, v_padding)

        # create Tensor datasets
        train_data_tensordt = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
        valid_data_tensordt = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

        # make sure to SHUFFLE your data
        train_loader = DataLoader(train_data_tensordt, shuffle=shuffle, batch_size=batch_size)
        valid_loader = DataLoader(valid_data_tensordt, shuffle=shuffle, batch_size=batch_size)

        return train_loader, valid_loader, vocab

