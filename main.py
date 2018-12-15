from data_preprocessing import preprocessing
from features_extraction import featureExtraction
from back_propagation import BP
from lda import LDA

def main():

    #preprocessing
    dataPre = preprocessing()

    combi = dataPre.read_data()

    combi = dataPre.clean_tweets(combi)

    #dataPre.visualize_data(combi)

    #dividing data



    #feature extraction
    fe = featureExtraction()




    #modeling BP


if __name__ == "__main__":
    main()