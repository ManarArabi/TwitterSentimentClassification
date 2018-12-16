
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_predict,cross_val_score
from data_preprocessing import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from svm import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#dataPre = preprocessing()
#dataPre.process_data()
#xtrain, xtest, ytrain, ytest = dataPre.divide_data()
#ytrain =ytrain.values.ravel()
#eclf = None

class Votting (object):
    # clf1 = svm.train(xtrain, ytrain)
    # clf2 = Bp.fit(xtrain, ytrain)

    #clf3 = LogisticRegression().fit(xtrain, ytrain)
    #clf4 = KNeighborsClassifier(5).fit(xtrain, ytrain)
    #clf5 = QuadraticDiscriminantAnalysis.fit(xtrain, ytrain)
    #clf6 = LinearDiscriminantAnalysis.fit(xtrain, ytrain)
    #clf7 = GaussianNB().fit(xtrain, ytrain)

    #estimators = [
     #'''('svm',clf1)'''   ,
      # ''' ('Bp', clf2)''',
       # ('log_reg',clf3),('knn',clf4)
        #,'''('lda',clf6)''',
        #'''('qda',clf5''',('rbf',clf7)]

    def __init__(self, xtrain, xtest, ytrain, ytest):
        print("Voting Systems")
        print("Gaussian NB VS. Logistic Regression  VS. KNN.")
        ytrain = ytrain.values.ravel()
        clf3 = LogisticRegression().fit(xtrain, ytrain)
        clf4 = KNeighborsClassifier(5).fit(xtrain, ytrain)
        clf7 = GaussianNB().fit(xtrain, ytrain)
        self.eclf = None
        self.estimators = [
            '''('svm',clf1)''',
            ''' ('Bp', clf2)''',
            ('log_reg', clf3), ('knn', clf4)
            , '''('lda',clf6)''',
            '''('qda',clf5''', ('rbf', clf7)]

    def votting(self, type):
        if type == 1:
            self.eclf =ensamble_hard = VotingClassifier(estimators=self.estimators, voting='hard')
        else:
            self.eclf =ensamble_soft = VotingClassifier(estimators=self.estimators, voting='soft')
        return self.eclf

    #
    # def get_accuracy(self,res,train ,test):
    #     acc = cross_val_score(res, train, test, cv=3)
    #     return acc




