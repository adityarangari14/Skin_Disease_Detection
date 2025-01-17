import tensorflow as tf
import numpy as np
import cv2
from keras.applications import  VGG19

def predict_skin_disease(image_path):
    # Define list of class names
    class_names = ["Acne","Eczema","Atopic","Psoriasis","Tinea","vitiligo"]


    vgg_model = VGG19(weights = 'imagenet',  include_top = False, input_shape = (180, 180, 3)) 
    # Load saved model
    model = tf.keras.models.load_model('./model/6claass.h5')

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = vgg_model.predict(img)
    img = img.reshape(1, -1)

    # Make prediction on preprocessed image
    pred = model.predict(img)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

print(predict_skin_disease("train/Atopic Dermatitis Photos/3IMG014.jpg"))



#/kaggle/input/dermnet/test/Atopic Dermatitis Photos/03ichthyosis050127.jpg
#C:\Users\Arya\OneDrive\Documents\ML\SEM-6\test\Atopic Dermatitis Photos\03ichthyosis050127.jpg