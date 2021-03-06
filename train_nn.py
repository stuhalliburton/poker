import os
import numpy as np
import pandas as pd
import operator

from keras.utils import to_categorical
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Flatten

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

def reshape_for_convolutional(data):
    return np.array(data).reshape(data.shape[0], 5, 17, 1)

class NeuralNetwork:
    MODEL_SAVE_PATH = './saved_models/model.h5'

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build()

    def _build(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 1), strides=1, input_shape=(5, 17, 1), activation='relu'))
        model.add(Conv2D(64, (1, 2), strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))

        # compile network
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train(self, x_train, y_train, batch_size=32, epochs=100,
            validation_split=0., validation_data=None):
        self.model.fit(x_train, y_train, epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save(self):
        # for layer in self.model.layers:
        #     print(layer.name)
        #     print(layer.get_weights())
        self.model.save(self.MODEL_SAVE_PATH)


if __name__ == '__main__':
    training_data_file = 'data/poker.data'
    test_data_file = 'data/poker-test.data'
    label_name = 'hand'
    num_classes = 10
    validation_split = 0.01
    epochs = 15
    batch_size = 32
    random_seed = None
    np.random.seed(random_seed)

    # load training data and format
    training_data = pd.read_csv(training_data_file).sample(frac=1)
    training_data = format_dataset(training_data)
    x_train = training_data.drop([label_name], 1)
    x_train = reshape_for_convolutional(x_train)
    y_train = training_data[label_name]
    y_train = to_categorical(y_train, num_classes)

    # load test/validation data and format
    test_data = pd.read_csv(test_data_file).sample(frac=1)
    test_data = format_dataset(test_data)
    test_features = test_data.drop([label_name], 1)
    test_features = reshape_for_convolutional(test_features)
    test_labels = test_data[label_name]
    test_labels = to_categorical(test_labels, num_classes)
    x_test, x_validation, y_test, y_validation = train_test_split(test_features,
            test_labels, test_size=validation_split, random_state=random_seed)

    # construct neural network
    network = NeuralNetwork(num_classes=num_classes)

    # train network on data
    network.train(x_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=[x_validation, y_validation])

    # evaluate performace on test data
    scores = network.evaluate(x_test, y_test)
    print 'Test Loss: {}, Test Accuracy {}'.format(scores[0], scores[1])

    # save model weights
    network.save()

    def find_test_hand(label):
        row = test_data.loc[test_data[label_name] == label][:1]
        row = row.drop([label_name], 1)
        row = reshape_for_convolutional(row)
        return row

    # load hands from test data
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
    predictions = network.predict(nothing)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Nothing - 0 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict one pair
    predictions = network.predict(one_pair)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'One Pair - 1 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict two pairs
    predictions = network.predict(two_pairs)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Two Pairs- 2 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict three of a kind
    predictions = network.predict(three_of_a_kind)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Three of a Kind - 3 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict straight
    predictions = network.predict(straight)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Straight - 4 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict flush
    predictions = network.predict(flush)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Flush - 5 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict full house
    predictions = network.predict(full_house)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Full House - 6 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict four of a kind
    predictions = network.predict(four_of_a_kind)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Four of a Kind - 7 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict straight flush
    predictions = network.predict(straight_flush)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Straight Flush - 8 : Prediction: {}, Confidence: {}'.format(prediction, confidence)

    # predict royal flush
    predictions = network.predict(royal_flush)
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Royal Flush - 9 : Prediction: {}, Confidence: {}'.format(prediction, confidence)
