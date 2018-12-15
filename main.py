from data_preprocessing import preprocessing

from back_propagation import BP
from lda import LDA
from qda import QDA
from svm import SVM




def main():

    #preprocessing
    dataPre = preprocessing()
    dataPre.process_data()
    xtrain, xtest, ytrain, ytest = dataPre.divide_data()

    #modeling (BP)
    my_model = BP()
    my_model.train(xtrain, ytrain)
    my_model.test(xtest)
    acc = my_model.get_accuracy(xtest, ytest)
    print(acc)

if __name__ == "__main__":
    main()