import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

trX, trY, teX, teY = mnist.load_data(one_hot=True, reshape=(-1, 1, 28, 28))

model = Sequential()
model.add(ZeroPadding2D((2,2), input_shape=trX.shape[1:]))

model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

num_epochs, batch_size, learn_rate = 30, 50, 0.1

model.compile(SGD(learn_rate), 'categorical_crossentropy', metrics=['accuracy'])
model.fit(trX, trY, batch_size, num_epochs, verbose=1, validation_data=(teX, teY))
