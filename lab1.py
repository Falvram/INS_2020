# from __future__ import absolute_import, division, print_function, unicode_literals
# import tensorflow as tf
#
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

# Тестирование при различных параметрах


def test_num_of_neurons():
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    vect_num_neurs = [4, 8, 12, 16]
    for i in vect_num_neurs:
        model = Sequential()
        model.add(Dense(4, activation='relu'))
        model.add(Dense(i, activation='relu'))
        model.add(Dense(i, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        H = model.fit(X, dummy_y, epochs=450, batch_size=10, validation_split=0.1)
        loss.append(H.history['loss'][-1])
        val_loss.append(H.history['val_loss'][-1])
        acc.append(H.history['accuracy'][-1])
        val_acc.append(H.history['val_accuracy'][-1])
    draw_test(vect_num_neurs, 'Number of neurons', loss, val_loss, acc, val_acc)


def test_num_of_layers():
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    vect_num_layers = [1, 2, 3]
    for i in vect_num_layers:
        model = Sequential()
        model.add(Dense(4, activation='relu'))
        for j in range(1, i):
            model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        H = model.fit(X, dummy_y, epochs=450, batch_size=10, validation_split=0.1)
        loss.append(H.history['loss'][-1])
        val_loss.append(H.history['val_loss'][-1])
        acc.append(H.history['accuracy'][-1])
        val_acc.append(H.history['val_accuracy'][-1])
    draw_test(vect_num_layers, 'Number of layers', loss, val_loss, acc, val_acc)


def test_epochs():
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    vect_epochs = range(50, 451, 50)
    for i in vect_epochs:
        model = Sequential()
        model.add(Dense(4, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        H = model.fit(X, dummy_y, epochs=i, batch_size=10, validation_split=0.1)
        loss.append(H.history['loss'][-1])
        val_loss.append(H.history['val_loss'][-1])
        acc.append(H.history['accuracy'][-1])
        val_acc.append(H.history['val_accuracy'][-1])
    draw_test(vect_epochs, 'Number of epochs', loss, val_loss, acc, val_acc)


def test_batch_size():
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    vect_batch = range(10, 80, 10)
    for i in vect_batch:
        model = Sequential()
        model.add(Dense(4, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        H = model.fit(X, dummy_y, epochs=450, batch_size=i, validation_split=0.1)
        loss.append(H.history['loss'][-1])
        val_loss.append(H.history['val_loss'][-1])
        acc.append(H.history['accuracy'][-1])
        val_acc.append(H.history['val_accuracy'][-1])
    draw_test(vect_batch, 'Batch size', loss, val_loss, acc, val_acc)


def validation_test():
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    vect_validation = []
    for i in range(1, 8):
        vect_validation.append(i*0.1)
        model = Sequential()
        model.add(Dense(4, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        H = model.fit(X, dummy_y, epochs=450, batch_size=10, validation_split=i*0.1)
        loss.append(H.history['loss'][-1])
        val_loss.append(H.history['val_loss'][-1])
        acc.append(H.history['accuracy'][-1])
        val_acc.append(H.history['val_accuracy'][-1])
    draw_test(vect_validation, 'Validation_split', loss, val_loss, acc, val_acc)


def draw_test(arg, label, loss, val_loss, acc, val_acc):
    plt.plot(arg, loss, 'r', label='Training loss')
    plt.plot(arg, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel(label)
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot(arg, acc, 'r', label='Training acc')
    plt.plot(arg, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel(label)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def best_model():
    model = Sequential()
    model.add(Dense(4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    H = model.fit(X, dummy_y, epochs=450, batch_size=10, validation_split=0.1)
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    print(len(loss))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


test_num_of_neurons()
test_num_of_layers()
test_epochs()
test_batch_size()
validation_test()
best_model()
