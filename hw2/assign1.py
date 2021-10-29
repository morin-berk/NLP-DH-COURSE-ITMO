from os import getcwd

import nltk
import numpy as np
from nltk.corpus import twitter_samples
from utils import build_freqs, process_tweet

nltk.download('twitter_samples')
nltk.download('stopwords')

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)),
                    np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)),
                   np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)


# ----- Logistic regression -----

def sigmoid(z):
    """
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    """
    h = 1/(1 + np.exp(-z))
    return h


def cost_func(h, m, y):
    tr_y = y.transpose()
    tr_y_one = (1 - y).transpose()
    J = np.mat(-1 / m) * (np.mat(tr_y) * np.mat(np.log(h)) +
                          np.mat(tr_y_one) * np.mat(np.log(1 - h)))
    return J


def gradientDescent(x, y, theta, alpha, num_iters):
    """
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    """
    m = len(x)

    for _ in range(0, num_iters):
        z = np.mat(x) * np.mat(theta)
        h = sigmoid(z)
        J = cost_func(h, m, y)
        theta = theta - (alpha/m)*(np.mat(x.transpose())*np.mat((h - y)))

    J = float(J)
    return J, theta


# ----- Extracting the features -----

def extract_features(tweet, freqs):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1

    for word in word_l:
        if (word, 1.0) in freqs:
            x[0, 1] += freqs.get((word, 1.0), 0)
        if (word, 0.0) in freqs:
            x[0, 2] += freqs.get((word, 0.0), 0)

    assert (x.shape == (1, 3))
    return x


# ----- Training Your Model -----

X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
# print(f"The cost after training is {J:.8f}.")
# print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")


def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.mat(x) * np.mat(theta))

    return y_pred


# ----- Test your logistic regression -----

def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    y_hat = []

    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0)

    y_hat_array = np.asarray(y_hat)
    test_y_array = np.squeeze(test_y)

    if y_hat_array == test_y_array:
        accuracy = sum(
            sum(y_hat_array),
            sum(test_y)/len(y_hat_array))
        return accuracy

# ---- Error Analysis ----

# print('Label Predicted Tweet')
# for x,y in zip(test_x,test_y):
#     y_hat = predict_tweet(x, freqs, theta)
#
#     if np.abs(y - (y_hat > 0.5)) > 0:
#         print('THE TWEET IS:', x)
#         print('THE PROCESSED TWEET IS:', process_tweet(x))
#         print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))


# ----- Predict with your own tweet ------

my_tweet = """imagine this: it's 4 am, you call an uber, 
              your Uber's name is "Stuart",
              you're waiting, it's says your Uber 
              is here but you see nothing, 
              you feel a nudge on your leg,
              you look down, there's a little red convertible, 
              it's your uber, your uber driver is Stuart Little"""

print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else:
    print('Negative sentiment')

# [[0.49843009]]
# Negative sentiment
