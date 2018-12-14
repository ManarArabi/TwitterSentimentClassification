from data_preprocessing import preprocessing
from back_propagation import BP
from lda import LDA

def main():

    #preprocessing
    dataPre = preprocessing()
    train_data, test_data, combi = dataPre.read_data()

    combi = dataPre.clean_tweets(combi)

    #modeling BP
    #model = BP('identity')
    #model = LDA()

    #model.train(train_data, test_data)

if __name__ == "__main__":
    main()