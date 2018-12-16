import pandas as pd
import scipy.sparse as sc
import numpy as np
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from visualization import VISUAL
from features_extraction import featureExtraction
import re


class preprocessing(object):

    train = None
    test = None
    combi = None
    fe = None

    features = None
    def __init__(self):

        self.fe = featureExtraction()
        return

    def remove_pattern(self, text, pattern):
        patterns = re.findall(pattern, text)
        for i in patterns:
            text = re.sub(i, '', text)

        return text

    def clean_tweets(self, data):
        temp = [None] * len(data['tweet'])

        for i in range(len(data['tweet'])):
            temp[i] = self.remove_pattern(data['tweet'][i], "@[\w]*")
            temp[i] = temp[i].replace("[^a-zA-Z#]", " ")
            line_string = ' '
            temp[i] = line_string.join([w for w in temp[i].split() if len(w) > 3])

        data['tidy_tweet'] = temp
        print("The tweets is cleaned.")
        return data

    def make_combination(self, train, test):
        combi = train.append(test, ignore_index=True)
        return combi

    def read_data(self):

        self.train = pd.read_csv('train_E6oV3lV.csv')
        self.test = pd.read_csv('test_tweets_anuFYb8.csv')
        self.combi = self.make_combination(self.train, self.test)
        print("The data is read.")

    def visualize_data(self):
        v = VISUAL(self.combi)
        print("Visualizing Data")
        v.visualize_data()

        return

    def slicing(self, combi):
        sliced = combi['tidy_tweet'].apply(lambda x: x.split())
        return sliced

    def stemming(self, data):
        stemmer = PorterStemmer()
        s = data.apply(lambda x: [stemmer.stem(i) for i in x])
        return s

    def process_data(self):
        self.read_data()
        self.combi = self.clean_tweets(self.combi)
        sliced = self.slicing(self.combi)
        sliced = self.stemming(sliced)
        for i in range(len(sliced)):
            sliced[i] = ' '.join(sliced[i])
        self.combi['tidy_tweet'] = sliced
        self.features = self.fe.get_bag_of_words(1000, 0.9, 2, self.combi, 'tidy_tweet')

    def divide_data(self):
        train = self.features[:31962, :]
        test = self.features[31962:, :]

        # splitting data into training and validation set
        xtrain, xtest, ytrain, ytest = train_test_split(train, self.train['label'], random_state=42, test_size=0.3)
        #print("________________")


        return xtrain, xtest, ytrain, ytest
