from data_preprocessing import preprocessing

from back_propagation import BP
from lda import LDA
from qda import QDA
from svm import SVM

import votting_networks

def main():

    #preprocessing
    dataPre = preprocessing()
    dataPre.process_data()
    xtrain, xtest, ytrain, ytest = dataPre.divide_data()

    #visualizing
    dataPre.visualize_data()

    #modeling (BP)
    print("Modeling ...")
    my_model = BP()
    my_model.train(xtrain, ytrain)
    #my_model.test(xtest)
    acc = my_model.get_accuracy(xtest, ytest)
    print("testing accuracy : ", "{0:.4f}".format(acc))
    print("*********************************************")
    #########################
    my_model = LDA()
    my_model.train(xtrain, ytrain)
    # my_model.test(xtest)
    acc = my_model.get_accuracy(xtest, ytest)
    print("testing accuracy : ", "{0:.4f}".format(acc))
    print("*********************************************")
    #########################
    my_model = QDA()
    my_model.train(xtrain, ytrain)
    # my_model.test(xtest)
    acc = my_model.get_accuracy(xtest, ytest)
    print("testing accuracy : ", "{0:.4f}".format(acc))
    print("*********************************************")
    #########################
    my_model = SVM()
    my_model.train(xtrain, ytrain)
    # my_model.test(xtest)
    acc = my_model.get_accuracy(xtest, ytest)
    print("testing accuracy : ", "{0:.4f}".format(acc))
    print("*********************************************")
    #########################

    # Votting classifiers
    #Votting = votting_networks.Votting(xtrain, xtest, ytrain, ytest)
    #res = Votting.votting(1)
    #print("voted " + str(res))

if __name__ == "__main__":
    main()