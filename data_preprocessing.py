import pandas as pd

import numpy as np

import re


class preprocessing(object):

    train = None
    test = None
    combi = None
    def __init__(self):
        #constructor
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
        return self.combi

    def visualize_data(self, data):
        print(data)
        return

