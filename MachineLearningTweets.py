



import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import matplotlib.pyplot as plt


def load_data(filename='training.small.csv'):
    reader = csv.reader(open(filename, 'r', encoding='utf-8', errors='ignore'))
    data = []
    for line in reader:
        data.append(line)
    return data

def train_vectorizer(tweets):
    cv = CountVectorizer(strip_accents='unicode', min_df=3)
    cv.fit(tweets)
    return cv

def vectorize_tweets(tweet_vectorizer, tweets):
    return tweet_vectorizer.transform(tweets)

def vectorize_sentiments(data):
    return np.array([int(row[0]) for row in data], dtype=np.int)
    

    

# Load raw data into nested lists
data = load_data()

# This code extracts only the tweets from the loaded data into a list of strings
tweets = [row[5] for row in data]

# Create a vectorizer - this will allow us to map each tweet to a vector of word counts
tweet_vectorizer = train_vectorizer(tweets)

# Test your tweet vectorizer using an example - note that the vectorize_tweets function
# takes a LIST OF TWEETS (strings), as a parameter and not a single tweet.
example_tweet = " hello from from the other side"
example_tweet_vector = vectorize_tweets(tweet_vectorizer, [example_tweet])

# Now apply the tweet vectorizer to all tweets - pay attention to the type and shape of what this function returns
tweet_vectors = vectorize_tweets(tweet_vectorizer, tweets)

# Get the sentiment labels from the data and organize it into a vector
sentiment_vector = vectorize_sentiments(data)


from scipy.sparse import vstack ##important for partitioning data

##code below partitions the data so that 90% of the dataset is used for training, while 10% is used for testing
##each set has equal amounts of positive and negative tweets

vector_training = vstack((tweet_vectors[0:72000],tweet_vectors[80000:152000]))  
vector_testing = vstack((tweet_vectors[72000:80000],tweet_vectors[152000:160000]))   
sentiment_training = np.array(sentiment_vector[0:72000].tolist() + sentiment_vector[80000:152000].tolist())
sentiment_testing = np.array(sentiment_vector[72000:80000].tolist() + sentiment_vector[152000:160000].tolist())



global svc              ##this allows changes that are made inside the function to be global
from sklearn.svm import LinearSVC       ##import LinearSVC Classification
svc = LinearSVC()

def learn1_svc():

   """
    Uses training data to train LinearSVC and then uses the testing data to see how accurate it is
    
   """
   
   svc.fit(vector_training,sentiment_training)      ##fit the training data of vector tweets and sentiments using LinearSVC
   correct = 0
   for i in range(vector_testing.shape[0]):             ##using the testing data, see how accurate LinearSVC is
       prediction = svc.predict(vector_testing[i])
       sentiment = sentiment_testing[i]
       if prediction[0] == sentiment:
           correct +=1
   accuracy = correct/vector_testing.shape[0]
   print('Linear Support Vector Classifier Testing Accuracy: {:.2f}'.format(accuracy))  ##print the accuracy of the algorithm
   
global sgd    ##allows changes that are made inside the function to be global
from sklearn.linear_model import SGDClassifier                          ##imports SGDC classifier
sgd = SGDClassifier(loss = "hinge", penalty = "l2", max_iter = 5)       
    
def learn2_sgd():

    """
    Uses training data to train SGD and then uses the testing data to see how accurate it is
    
    """

    sgd.fit(vector_training,sentiment_training)     ##fits the training data of vector tweets and sentiments using SGDClassifier
    correct = 0
    for i in range(vector_testing.shape[0]):        ##using the testing, data see how accurate SGDC is
        prediction = sgd.predict(vector_testing[i])
        sentiment = sentiment_testing[i]
        if prediction[0] == sentiment:
            correct +=1
            
    accuracy = correct/vector_testing.shape[0]
    print('Stochastic Gradient Descent Classifier Testing Accuracy: {:.2f}'.format(accuracy))   ##prints the accuracy of the algorithm

        
def predict_sentiment(tweet_vectorizer, my_model, tweet):

    """
        Predicts the sentiment as 'positive' or 'negative' of a new tweet
        If using svc as model, learn1_svc() needs to be run first in order to fit the data to the model
        same goes with sgd, learn2_sgd() must be run first if using sgd as a model
        
        param: tweet_vectorizer
        param: my_model: name of machine learning algorithm ex. sgd, svc
        param: tweets: new tweet as a string
    """
    
    test_tweet_vectors = vectorize_tweets(tweet_vectorizer, [tweet])   ##first vectorize your new tweet
    test_tweet_sentiments = my_model.predict(test_tweet_vectors)        ##use your machine learning model to predict the sentiment
    for i in test_tweet_sentiments:     
        if i == 0:
            print('Negative')
        elif i == 4:
            print('Positive')
 

