from sklearn.neural_network import MLPClassifier


class BP(object):

    clf = None

    def __init__(self, activationF):
        #solver = {‘lbfgs’, ‘sgd’, ‘adam’}
        #activationF = {‘identity’, ‘logistic’, ‘tanh’}
        #default is : 100 neuron with one hidden layer
        #to change : hidden_layer_sizes=(5, 2)
        self.clf = MLPClassifier(solver='lbfgs', activation=activationF)

    def train(self, x, y):
        self.clf.fit(x, y)

    def test(self, x):
        y = self.clf.predict(x)
        return y

    def get_accuracy(self, test_data, test_res):
        acc = self.clf.score(test_data, test_res)
        return acc
