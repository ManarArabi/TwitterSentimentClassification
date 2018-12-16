from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class BP(object):

    clf = None

    def __init__(self, activationF = 'identity'):
        #solver = {‘lbfgs’, ‘sgd’, ‘adam’}
        #activationF = {‘identity’, ‘logistic’, ‘tanh’}
        #default is : 100 neuron with one hidden layer
        #to change : hidden_layer_sizes=(5, 2)
        #self.clf = MLPClassifier(solver='lbfgs', activation=activationF)
        mlp = MLPClassifier(max_iter=1000, random_state=0)
        self.parameters_grid = [
            {
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'hidden_layer_sizes': [
                    (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,)
                ]
            }
        ]
        self.clf = GridSearchCV(mlp, self.parameters_grid, cv=3, scoring='accuracy')


    def train(self, x, y):
        print("training ...")
        self.clf.fit(x, y)
        print(self.clf.best_params_())

    def test(self, x):
        y = self.clf.predict(x)
        return y

    def get_accuracy(self, test_data, test_res):
        print("testing ...")
        acc = self.clf.score(test_data, test_res)
        return acc
