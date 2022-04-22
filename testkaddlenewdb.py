import tensorflow as tf
train_dir ='/Users/nicolomorando/Downloads/dataset/train'
test_dir  ='/Users/nicolomorando/Downloads/dataset/test'
width, height = 86, 86
training=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,
                                                          rotation_range=7,
                                                          horizontal_flip=True,
                                                          validation_split=0.05
                                                         ).flow_from_directory(train_dir,
                                                                               class_mode = 'categorical',
                                                                               batch_size = 8,
                                                           target_size=(width,height),
                                                                              subset="training")
testing=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,
                                                         ).flow_from_directory(test_dir,
                                                                               class_mode = 'categorical',
                                                                               batch_size = 8,
                                                                               shuffle = False,
                                                           target_size=(width,height))
validing=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,
                                                          rotation_range=7,
                                                          horizontal_flip=True,
                                                         validation_split=0.05
                                                        ).flow_from_directory(train_dir,
                                                                              batch_size = 8,
                                                                              class_mode = 'categorical',
                                                           target_size=(width,height),subset='validation',shuffle=True)

from keras.models import Sequential ,Model
from keras.layers import Dense ,Flatten ,Conv2D ,MaxPooling2D ,Dropout ,BatchNormalization  ,Activation ,GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau

optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.99,decay=0.001/32)
EarlyStop=EarlyStopping(patience=10,restore_best_weights=True)
Reduce_LR=ReduceLROnPlateau(monitor='val_accuracy',verbose=2,factor=0.5,min_lr=0.00001)
callback=[EarlyStop , Reduce_LR]
num_classes = 2
num_detectors=32

network = Sequential()

network.add(Conv2D(num_detectors, (3,3), activation='relu', padding = 'same', input_shape = (width, height, 3)))
network.add(BatchNormalization())
network.add(Conv2D(num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(Conv2D(2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Flatten())

network.add(Dense(2 * num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(2 * num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(num_classes, activation='softmax'))
network.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=["accuracy"])
history=network.fit(training,validation_data=validing,epochs=20, callbacks=callback, verbose=2)
network.save_weights('model_weight.h5')
network.save('blinking_cnn_model.h5')
