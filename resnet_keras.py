import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.optimizers import SGD

def residual_block(nb_filter, nb_row, nb_col, input_shape, n_skip=2):
    x = Input(shape=(input_shape))
    y = x
    for i in range(n_skip):
        y = BatchNormalization(axis=1)(y)
        y = Activation('relu')(y)
        y = Convolution2D(nb_filter, nb_row, nb_col, border_mode='same')(y)
    y = merge([x, y], mode='sum')
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    return Model(input=x, output=y)

trX, trY, teX, teY = mnist.load_data(one_hot=True, reshape=(-1, 1, 28, 28))

model = Sequential()
model.add(Convolution2D(16, 3, 3, input_shape=trX.shape[1:], border_mode='same', activation='relu'))
model.add(residual_block(16, 3, 3, input_shape=model.output_shape[1:]))
model.add(residual_block(16, 3, 3, input_shape=model.output_shape[1:]))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

num_epochs, batch_size, learn_rate = 12, 20, 0.1

model.compile(SGD(learn_rate), 'categorical_crossentropy', metrics=['accuracy'])
model.fit(trX, trY, batch_size, num_epochs, verbose=1, validation_data=(teX, teY))
