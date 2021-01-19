import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(picture):
    loaded_model = tf.keras.models.load_model('./saved/model.h5')
    picture = image.load_img(picture, target_size = (299,299))
    picture = image.img_to_array(picture)
    picture = np.expand_dims(picture, axis=0)
    picture /= 255.

    prediction = loaded_model.predict(picture)
    index = np.argmax(prediction)

    food_classes = ['pancackes', 'donuts', 'pizza', 'pho', 'hamburger']
    food_classes.sort()

    return food_classes[index]