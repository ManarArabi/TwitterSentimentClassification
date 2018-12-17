# TwitterSentimentClassification
classify tweets to positive or negative tweets.


## Task

Detecting hate speech in tweets. 

For the sake of simplicity, we say a tweet contains hate speech if it has a good or bad sentiment associated with it. So, the task is to classify good or bad tweets from other tweets.



## Solution 
We choose common Techniques in neural networks field to get the greatest accuracy. 

1. Back Propagation Model (1 hidden layer, 100 neuron)
      
2. Linear discriminant analysis 

3. Quadratic discriminant analysis 

4. Support vector machine (RBF kernel function)

5. voting between other models (KNN, Gaussian NB, Logistic regression) 

##Steps 

* Clean Tweets .
    
    
    removing @user, hashtag, etc...
* Extract features from it to use it as input in the network.
    
    
    using Bag of words technique or TF-IDF 
*  Pass input to each model and compare accuracy.

##Result 

Back probagation has the greatest accuracy (0.946).

####notes
If you aren't familiar with the above networks you should check the links below.

* Back propagation : 


    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    
* LDA & QDA: 


    http://uc-r.github.io/discriminant_analysis
    
* SVM :


    https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989
    


    
 
