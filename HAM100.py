import tensorflow as tf
import numpy as np
import cv2
import os
#from google.colab.patches import cv2_imshow

def predict_cancer_class(image_path):
    # Define list of class names
    classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}

    model = tf.keras.models.load_model('./model/best_model.h5')

    img = cv2.imread(image_path)

    # Display the image
    #cv2_imshow(img)

    # Resize the image to (28, 28)
    img = cv2.resize(img, (28, 28))

    # Make prediction using the pre-trained model (assuming 'model' and 'classes' are defined elsewhere)
    result = model.predict(img.reshape(1, 28, 28, 3))

    # Find the maximum probability and corresponding class index
    max_prob = max(result[0])
    class_ind = list(result[0]).index(max_prob)

    # Get the class name from the list 'classes'
    class_name = classes[class_ind]

    # Print the predicted class name
    print(class_name)
        
    return class_name


print(predict_cancer_class('../HAM100/HAM10000_images_part_1/ISIC_0024743.jpg'))