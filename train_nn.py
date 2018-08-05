import os
import numpy as np
import pandas as pd

from keras.utils import to_categorical
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

def format_dataset(data):
    heart_lambda   = lambda x: 1 if x == 1 else 0
    spade_lambda   = lambda x: 1 if x == 2 else 0
    diamond_lambda = lambda x: 1 if x == 3 else 0
    club_lambda    = lambda x: 1 if x == 4 else 0

    data['s1_h'] = data['s1'].apply(heart_lambda)
    data['s1_s'] = data['s1'].apply(spade_lambda)
    data['s1_d'] = data['s1'].apply(diamond_lambda)
    data['s1_c'] = data['s1'].apply(club_lambda)

    data['s2_h'] = data['s2'].apply(heart_lambda)
    data['s2_s'] = data['s2'].apply(spade_lambda)
    data['s2_d'] = data['s2'].apply(diamond_lambda)
    data['s2_c'] = data['s2'].apply(club_lambda)

    data['s3_h'] = data['s3'].apply(heart_lambda)
    data['s3_s'] = data['s3'].apply(spade_lambda)
    data['s3_d'] = data['s3'].apply(diamond_lambda)
    data['s3_c'] = data['s3'].apply(club_lambda)

    data['s4_h'] = data['s4'].apply(heart_lambda)
    data['s4_s'] = data['s4'].apply(spade_lambda)
    data['s4_d'] = data['s4'].apply(diamond_lambda)
    data['s4_c'] = data['s4'].apply(club_lambda)

    data['s5_h'] = data['s5'].apply(heart_lambda)
    data['s5_s'] = data['s5'].apply(spade_lambda)
    data['s5_d'] = data['s5'].apply(diamond_lambda)
    data['s5_c'] = data['s5'].apply(club_lambda)

    return data.drop(['s1', 's2', 's3', 's4' ,'s5'], 1)

class NeuralNetwork:
    MODEL_SAVE_PATH = './saved_models/model.h5'

    def __init__(self, feature_count=25, num_classes=10):
        self.feature_count = feature_count
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build()

    def _build(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.feature_count, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train(self, x_train, y_train, batch_size=32, epochs=100,
            validation_data=None):
        self.model.fit(x_train, y_train, epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save(self):
        self.model.save(self.MODEL_SAVE_PATH)


if __name__ == '__main__':
    data_file = 'data/poker.data'
    label_name = 'hand'
    feature_count = 25
    num_classes = 10
    test_ratio = 0.1
    random_seed = None
    epochs = 1000
    batch_size = 50

    # load CSV and format
    dataset = pd.read_csv(data_file)
    dataset = format_dataset(dataset)

    # split features & labels
    features = dataset.drop([label_name], 1)
    labels = dataset[label_name]

    # categorise labels
    labels = to_categorical(labels, num_classes)

    # split training from test data
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_ratio, random_state=random_seed)

    network = NeuralNetwork(feature_count=feature_count, num_classes=num_classes)

    # train network on data
    network.train(x_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=[x_test, y_test])

    # evaluate performace on test data
    scores = network.evaluate(x_test, y_test)
    print 'Test Loss: {}, Test Accuracy {}'.format(scores[0], scores[1])

    # save model weights
    network.save()


    # create test dataframe - 7 : four of a kind
    four_of_a_kind = pd.DataFrame()
    four_of_a_kind['s1'] = [1]
    four_of_a_kind['c1'] = [7]
    four_of_a_kind['s2'] = [2]
    four_of_a_kind['c2'] = [7]
    four_of_a_kind['s3'] = [3]
    four_of_a_kind['c3'] = [7]
    four_of_a_kind['s4'] = [4]
    four_of_a_kind['c4'] = [7]
    four_of_a_kind['s5'] = [1]
    four_of_a_kind['c5'] = [12]
    four_of_a_kind = format_dataset(four_of_a_kind)

    # predict four of a kind
    predictions = network.predict(four_of_a_kind)

    import operator
    (prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    print 'Prediction: {}, Confidence: {}'.format(prediction, confidence)
