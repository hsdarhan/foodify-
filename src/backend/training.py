import tensorflow as tf
import helper

from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD

# Separate files into training and testing directories
helper.separate_files('food-101/images', 'food-101/testing', 'food-101/meta/test.txt')
helper.separate_files('food-101/images', 'food-101/training', 'food-101/meta/train.txt')

# Limiting to the following classes:
food_classes = ['pancackes', 'donuts', 'pizza', 'pho', 'hamburger']

# Isolate the classes
helper.copy_repository('food-101/training', 'food-101/training-model', food_classes)
helper.copy_repository('food-101/testing', 'food-101/testing-model', food_classes)

num_classes = len(food_classes)
img_width, img_height = 299, 299
training_set = 'food-101/training-model'
testing_set = 'food-101/testing-model'
num_training_images = 3750
num_testing_images = 1250
batch_size = 16

training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

testing_data_generator = ImageDataGenerator(rescale=1. / 255)

training_generator = training_data_generator.flow_from_directory(
    training_set,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

testing_generator = testing_data_generator.flow_from_directory(
    testing_set,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

predictions = tf.keras.layers.Dense(num_classes, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='best_model.h5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history.log')

history = model.fit_generator(training_generator,
                              steps_per_epoch=num_training_images // batch_size,
                              validation_data=testing_generator,
                              validation_steps=num_testing_images // batch_size,
                              epochs=10,
                              verbose=1,
                              callbacks=[csv_logger, checkpoint])

model.save('./saved/model.h5')