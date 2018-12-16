from sklearn import svm


class SVM(object):

    clf = None

    def __init__(self, kernelF = 'rbf'):
        #kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’} defualt =rbf
        print("SVM Model")
        self.clf = svm.SVC(gamma='scale', kernel=kernelF)
        return

    def train(self, x, y):
        print("Training ...")
        self.clf.fit(x, y)

    def test(self, x):
        y = self.clf.predict(x)
        return y

    def get_accuracy(self, test_data, test_res):
        print("testing ...")
        acc = self.clf.score(test_data, test_res)
        return acc
