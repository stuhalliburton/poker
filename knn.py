import numpy as np
import pandas as pd

from keras.utils import to_categorical
from sklearn import neighbors
from sklearn.model_selection import train_test_split

training_data_file = 'data/poker.data'
test_data_file = 'data/poker-test.data'
label_name = 'hand'
num_classes = 10

def format_dataset(data):
    for card in range(1, 6, 1):
        for value in range(1, 14, 1):
            card_header = 'c{}_{}'.format(card, value)
            card_number = 'c{}'.format(card)
            card_lambda = lambda x: 1 if x == value else 0
            data[card_header] = data[card_number].apply(card_lambda)

        for index, suit in enumerate(['h', 's', 'd', 'c']):
            suit_header = 's{}_{}'.format(card, suit)
            suit_number = 's{}'.format(card)
            suit_lambda = lambda x: 1 if x == index+1 else 0
            data[suit_header] = data[suit_number].apply(suit_lambda)

    return data.drop(['c1', 'c2', 'c3', 'c4', 'c5', 's1', 's2', 's3', 's4' ,'s5'], 1)

# load training data and format
print 'loading training data...'
training_data = pd.read_csv(training_data_file)
training_data = format_dataset(training_data)
x_train = training_data.drop([label_name], 1)
y_train = training_data[label_name]
y_train = to_categorical(y_train, num_classes)
print 'done.'

# load test/validation data and format
print 'loading test/validation data...'
test_data = pd.read_csv(test_data_file)
test_data = format_dataset(test_data)
x_test = test_data.drop([label_name], 1)
y_test = test_data[label_name]
y_test = to_categorical(y_test, num_classes)
print 'done.'

# define and train classifier
print 'training classifier...'
knn = neighbors.KNeighborsClassifier(n_neighbors=3, p=2)
knn.fit(x_train, y_train)
print 'done.'

# evaluate classifier
print 'evaluating classifier...'
accuracy = knn.score(x_test[:2000], y_test[:2000])
print 'Accuracy: {}'.format(accuracy)
print 'done.'


# load hands from test data
find_test_hand = lambda x: test_data.loc[test_data[label_name] == x][:1].drop([label_name], 1)
nothing = find_test_hand(0)
one_pair = find_test_hand(1)
two_pairs = find_test_hand(2)
three_of_a_kind = find_test_hand(3)
straight = find_test_hand(4)
flush = find_test_hand(5)
full_house = find_test_hand(6)
four_of_a_kind = find_test_hand(7)
straight_flush = find_test_hand(8)
royal_flush = find_test_hand(9)

# predict nothing
prediction = knn.predict(nothing)
print 'Nothing - 0 : Prediction: {}'.format(prediction)

# predict one pair
prediction = knn.predict(one_pair)
print 'One Pair - 1 : Prediction: {}'.format(prediction)

# predict two pairs
prediction = knn.predict(two_pairs)
print 'Two Pairs- 2 : Prediction: {}'.format(prediction)

# predict three of a kind
prediction = knn.predict(three_of_a_kind)
print 'Three of a Kind - 3 : Prediction: {}'.format(prediction)

# predict straight
prediction = knn.predict(straight)
print 'Straight - 4 : Prediction: {}'.format(prediction)

# predict flush
prediction = knn.predict(flush)
print 'Flush - 5 : Prediction: {}'.format(prediction)

# predict full house
prediction = knn.predict(full_house)
print 'Full House - 6 : Prediction: {}'.format(prediction)

# predict four of a kind
prediction = knn.predict(four_of_a_kind)
print 'Four of a Kind - 7 : Prediction: {}'.format(prediction)

# predict straight flush
prediction = knn.predict(straight_flush)
print 'Straight Flush - 8 : Prediction: {}'.format(prediction)

# predict royal flush
prediction = knn.predict(royal_flush)
print 'Royal Flush - 9 : Prediction: {}'.format(prediction)
