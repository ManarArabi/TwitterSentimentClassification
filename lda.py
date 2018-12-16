from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA(object):
    clf = None

    def __init__(self):
        print("LDA Model")
        self.clf = LinearDiscriminantAnalysis()

    def train(self, x, y):
        print("Training ...")
        self.clf.fit(x, y)

    def test(self, x):
        res = self.clf.predict(x)
        return res

    def get_accuracy(self, test_data, test_res):
        print("testing ...")
        acc = self.clf.score(test_data, test_res)
        return acc
