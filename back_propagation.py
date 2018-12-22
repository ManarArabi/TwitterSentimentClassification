from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class BP(object):

    clf = None

    def __init__(self, activationF = 'tanh'):
        #solver = {‘lbfgs’, ‘sgd’, ‘adam’}
        #activationF = {‘identity’, ‘logistic’, ‘tanh’}
        #default is : 100 neuron with one hidden layer
        #to change : hidden_layer_sizes=(5, 2)
        print("Back propagation Model")
        self.clf = MLPClassifier(solver='sgd', activation=activationF, learning_rate='adaptive')


    def train(self, x, y):
        print("training ...")
        self.clf.fit(x, y)

    def test(self, x):
        y = self.clf.predict(x)
        return y

    def get_accuracy(self, test_data, test_res):
        print("testing ...")
        acc = self.clf.score(test_data, test_res)
        return acc
