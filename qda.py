from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class QDA(object):
    clf = None

    def __init__(self):
        self.clf = QuadraticDiscriminantAnalysis()

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