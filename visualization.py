import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud


class VISUAL(object):
    def __init__(self, data):
        self.data = data

    def __commonWord__(self):
        all_words = ' '.join([text for text in  self.data['tidy_tweet']])
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
        #plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

        return all_words

    def __normalWords__(self):
        normal_words = ' '.join([text for text in self.data['tidy_tweet'][self.data['label'] == 0]])
        wordcloud = WordCloud(width=800, height=500,
                              random_state=21, max_font_size=110).generate(normal_words)
        #plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

        return normal_words

    def __negativeWords__(self):
        negative_words = ' '.join([text for text in self.data['tidy_tweet'][self.data['label'] == 1]])
        wordcloud = WordCloud(width=800, height=500,
                              random_state=21, max_font_size=110).generate(negative_words)
        #plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

        return negative_words

    def visulizeWords(self):
        self.__commonWord__()
        self.__normalWords__()
        self.__negativeWords__()

    def __hashtagExtract__(self, inputText):
        hashtags = []
        # Loop over the words in the tweet
        for i in inputText:
            ht = re.findall(r"#(\w+)", i)
            hashtags.append(ht)

        return hashtags

    def hashtagList(self):
        HT_regular = self.__hashtagExtract__(self.data['tidy_tweet'][self.data['label'] == 0])
        HT_negative = self.__hashtagExtract__(self.data['tidy_tweet'][self.data['label'] == 1])
        HT_regular = sum(HT_regular, [])
        HT_negative = sum(HT_negative, [])
        return HT_regular, HT_negative

    def plotHashtags(self,input):
        a = nltk.FreqDist(input)
        d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
        d = d.nlargest(columns="Count", n = 10)
        ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
        ax.set(ylabel = 'Count')
        plt.show()

    def visualize_data(self):
        #add print here to know what is visualized in each figure
        self.visulizeWords()
        regulare, hate = self.hashtagList()
        self.plotHashtags(regulare)
        self.plotHashtags(hate)

