from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.utils import np_utils
from keras import backend as K
from scipy import io as sio

import dataset_path
import parameters

print('backend', K.backend())


#[Dataset taken from : https://www.nist.gov/itl/iad/image-group/emnist-dataset]

'''
Load dataset in matlab format
'''
def load_input(input_index):
    mat = sio.loadmat(dataset_path.data_path[input_index])
    data = mat['dataset']
    return data

'''
Extract training and test images and labels
'''
def extract_train_test(data):
    X_train = data['train'][0, 0]['images'][0, 0]
    y_train = data['train'][0, 0]['labels'][0, 0]
    X_test = data['test'][0, 0]['images'][0, 0]
    y_test = data['test'][0, 0]['labels'][0, 0]
    return X_train, y_train, X_test, y_test

'''
Reshape data to match the number of rows and cols in images
'''
def reshape_data(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], 28, 28, order="A")
    X_test = X_test.reshape(X_test.shape[0], 28, 28, order="A")
    return X_train, X_test

'''
Convert data to binary format matrix
'''
def convert_data(X_train, X_test, y_train, y_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, parameters.no_classes)
    Y_test = np_utils.to_categorical(y_test, parameters.no_classes)
    return X_train, X_test, Y_train, Y_test

'''
Print utility to print the data stats
'''
def print_data_stats(X_train, y_train, X_test, Y_train):
    print()
    print('MNIST data loaded: train:', len(X_train), 'test:', len(X_test))
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('Y_train:', Y_train.shape)

if __name__ == '__main__':

    data = load_input(0)
    X_train, y_train, X_test, y_test = extract_train_test(data)
    X_train, X_test = reshape_data(X_train, X_test)
    X_train, X_test, Y_train, Y_test = convert_data(X_train, X_test, y_train, y_test)
    model = Sequential()
    model.add(SimpleRNN(parameters.no_units,
                        input_shape=(parameters.no_rows, parameters.no_cols)))
    model.add(Dense(units=parameters.no_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    epochs = 100
    history = model.fit(X_train,
                        Y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2)

    scores = model.evaluate(X_test, Y_test, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
