import tensorflow as tf
import pandas
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import numpy as np

# builds 4 models with different architectures
def get_models():
    # basic model
    model1 = Sequential()
    model1.add(Dense(16, activation='relu', input_shape=(16, )))
    model1.add(Dense(7, activation='softmax'))

    # basic with dropout layer
    model2 = Sequential()
    model2.add(Dense(16, activation='relu', input_shape=(16, )))
    model2.add(Dropout(0.25))
    model2.add(Dense(7, activation='softmax'))

    # basic with extra  dense layer
    model2 = Sequential()
    model2.add(Dense(16, activation='relu', input_shape=(16, )))
    model2.add(Dense(32, activation='relu'))
    model2.add(Dense(7, activation='softmax'))

    # basic + dropout + extra dense layer
    model3 = Sequential()
    model3.add(Dense(16, activation='relu', input_shape=(16, )))
    model3.add(Dropout(0.25))
    model3.add(Dense(32, activation='relu'))
    model3.add(Dropout(0.25))
    model3.add(Dense(7, activation='softmax'))

     # basic + dropouts + 2 extra dense layers
    model4 = Sequential()
    model4.add(Dense(16, activation='relu', input_shape=(16, )))
    model4.add(Dense(32, activation='relu'))
    model4.add(Dropout(0.25))
    model4.add(Dense(32, activation='relu'))
    model4.add(Dropout(0.25))
    model4.add(Dense(7, activation='softmax'))

    return [model1, model2, model3, model4]

def build_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(16, )))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(7, activation='softmax'))
    return model

def load_data(filename):
    dataframe = pandas.read_csv(filename, header=None)
    dataset = dataframe.values
    X = dataset[:,1:17].astype(float)   # animal data
    Y = dataset[:,17].astype(float)     # animal types
    return np.array(X), np.array(Y)

def split_data(X, Y, val_split):
    assert(len(X) == len(Y))
    border = int(np.floor(len(X) * (1 - val_split)))
    training_data = X[0:border]
    testing_data = X[border:]
    training_targets = Y[0:border]
    testing_targets = Y[border:]
    return (training_data, training_targets), (testing_data, testing_targets)

def save_plot(label, history):
    picdir = './pics/'
    plt.clf()
    arg = range(1, len(history['loss']) + 1)

    # save accuracy
    plt.clf()
    plt.plot(arg, history['acc'], 'r', label='Training accuracy')
    plt.plot(arg, history['val_acc'], 'b', label='Validation accuracy')
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(picdir + label + '_accuracy' + '.png')

    # save loss
    plt.clf()
    plt.plot(arg, history['loss'], 'r', label='Training loss')
    plt.plot(arg, history['val_loss'], 'b', label='Validation loss')
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(picdir + label + '_loss' + '.png')

def explore_optimizer(optimizer, label):
    model = build_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    H = model.fit(training_data, training_targets, epochs=750, batch_size=10,
            validation_data=(testing_data, testing_targets),
            callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience = 100),
                        EarlyStopping(monitor='val_acc', mode='max', patience = 100)])
    save_plot(label, H.history)

def prepare_targets(targets):
    targets = [t - 1 for t in targets] # cuz labels are ints of range [1:7]
    return to_categorical(targets)

def try_some_optimizers():
    rate = [0.1, 0.01, 0.001]
    momentum = [0, 0.1, 0.5, 0.95]
    rho = [0, 0.9, 0.95, 0.99, 10]
    for r in rate:
        for m in momentum:
            explore_optimizer(optimizers.SGD(learning_rate=r, momentum=m), 'SGD_rate='+str(r)+'_momentum='+str(m))
        explore_optimizer(optimizers.Adam(learning_rate=r), 'Adam_rate='+str(r))
        # It is recommended to leave the parameters of this optimizer at their default values
        # (except the learning rate, which can be freely tuned).
        explore_optimizer(optimizers.RMSprop(learning_rate=r), 'RMSprop_rate='+str(r))
    for r in rho:
        # Initial learning rate is recommended to leave at the default value.
        explore_optimizer(optimizers.Adadelta(rho=r), 'Adadelta_rho='+str(r))

def try_some_models(models):
    for i, model in enumerate(models):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        H = model.fit(training_data, training_targets, epochs=750, batch_size=10,
            validation_data=(testing_data, testing_targets),
            callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience = 100),
                        EarlyStopping(monitor='val_acc', mode='max', patience = 100)])
        save_plot('model'+str(i), H.history)


# load and prepare the data
X, Y = load_data("zoo.data")
Y = prepare_targets(Y)
(training_data, training_targets), (testing_data, testing_targets) = split_data(X, Y, 0.1)

# try some models in order to choose the best one
models = get_models()
try_some_models(models)

# get the model chosen and try some optimizers in order to choose the one that fits best
model = build_model()
try_some_optimizers()



