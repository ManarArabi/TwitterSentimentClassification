from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import numpy as np


class featureExtraction(object):
    def __init__(self):
        return

    def get_bag_of_words(self, maxi_features, max_freq, min_freq, data, column_name):
        # Count Vectorizer
        bow_vectorizer = CountVectorizer(
            max_df=max_freq,
            min_df=min_freq,
            max_features=maxi_features,
            stop_words='english'
        )
        ## max_df ignore all words which has frequency more than the threshold
        ## min_df ignore all words which has frequency less than the threshold
        ## max_features build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
        ## stop_words Terms that were ignored because they either: max_dif or min_dif or max_feature

        bow = bow_vectorizer.fit_transform(data[column_name]).toarray().astype(np.float32)

        #print(type(bow))
        # the output is the word id , number of occurrence , weighting with diminishing importance
        return bow


    def ti_idf(self, maxi_features, max_freq, min_freq, data, column_name):
        tfidf_vectorizer = TfidfVectorizer(
            max_df=max_freq,
            min_df=min_freq,
            max_features=maxi_features,
            stop_words='english'
        )

        tfidf = tfidf_vectorizer.fit_transform(data[column_name])

        return tfidf

    # links :
    # https://scikit-learn.org/stable/modules/feature_extraction.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html



