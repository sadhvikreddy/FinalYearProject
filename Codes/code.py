#for predicting

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
import keras
from Website.Codes import db


def compute(fname):
    keras.backend.clear_session()

    classifier = Sequential()

    # Convolution Step 1
    classifier.add(Convolution2D(96, 11, strides=(4, 4), padding='valid', input_shape=(224, 224, 3), activation='relu'))

    # Max Pooling Step 1
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 2
    classifier.add(Convolution2D(256, 11, strides=(1, 1), padding='valid', activation='relu'))

    # Max Pooling Step 2
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 3
    classifier.add(Convolution2D(384, 3, strides=(1, 1), padding='valid', activation='relu'))
    classifier.add(BatchNormalization())

    # Convolution Step 4
    classifier.add(Convolution2D(384, 3, strides=(1, 1), padding='valid', activation='relu'))
    classifier.add(BatchNormalization())

    # Convolution Step 5
    classifier.add(Convolution2D(256, 3, strides=(1, 1), padding='valid', activation='relu'))

    # Max Pooling Step 3
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Flattening Step
    classifier.add(Flatten())

    # Full Connection Step
    classifier.add(Dense(units=4096, activation='relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=4096, activation='relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=1000, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=38, activation='softmax'))

    from keras import optimizers
    classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    '''
    li = ['Apple___Apple_scab',
          'Apple___Black_rot',
          'Apple___Cedar_apple_rust',
          'Apple___healthy',
          'Blueberry___healthy',
          'Cherry_(including_sour)___Powdery_mildew',
          'Cherry_(including_sour)___healthy',
          'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
          'Corn_(maize)___Common_rust_',
          'Corn_(maize)___Northern_Leaf_Blight',
          'Corn_(maize)___healthy',
          'Grape___Black_rot',
          'Grape___Esca_(Black_Measles)',
          'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
          'Grape___healthy',
          'Orange___Haunglongbing_(Citrus_greening)',
          'Peach___Bacterial_spot',
          'Peach___healthy',
          'Pepper,_bell___Bacterial_spot',
          'Pepper,_bell___healthy',
          'Potato___Early_blight',
          'Potato___Late_blight',
          'Potato___healthy',
          'Raspberry___healthy',
          'Soybean___healthy',
          'Squash___Powdery_mildew',
          'Strawberry___Leaf_scorch',
          'Strawberry___healthy',
          'Tomato___Bacterial_spot',
          'Tomato___Early_blight',
          'Tomato___Late_blight',
          'Tomato___Leaf_Mold',
          'Tomato___Septoria_leaf_spot',
          'Tomato___Spider_mites Two-spotted_spider_mite',
          'Tomato___Target_Spot',
          'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
          'Tomato___Tomato_mosaic_virus',
          'Tomato___healthy']
    '''

    plant,disease, li = db.send_list()

    filepath = "/Users/sadhvik/Desktop/FinalProject/output/finalweights.hdf5"
    classifier.load_weights(filepath)
    image_path = "/Users/sadhvik/Desktop/FinalProject/Website/static/images/Unprocessed/" + fname
    new_img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    #predicting
    prediction = classifier.predict(img)
    d = prediction.flatten()
    j = d.max()
    for index, item in enumerate(d):
        if item == j:
            class_name = li[index]
            plant_name = plant[index]
            disease_name = disease[index]

    sym,tre = db.get_sym_tre(plant_name,disease_name)


    acc = round(j, 2)
    if (acc >= 0.50):
        #sending output on success
        acc = "Confidence: " + str(acc)
        print(class_name)
        plt.figure(figsize=(4, 4))
        plt.imshow(new_img)
        plt.axis('off')
        plt.title(class_name)
        plt.text(75, 250, acc)
        image_path = "/Users/sadhvik/Desktop/FinalProject/Website/static/images/Processed/" + fname
        plt.savefig(image_path)
        return plant_name,disease_name,sym,tre
    else:
        #sending output on failure
        image_path = "/Users/sadhvik/Desktop/FinalProject/Website/static/images/Processed/" + fname
        plt.figure(figsize=(4, 4))
        plt.imshow(new_img)
        plt.axis('off')
        plt.text(75, 250, "No leaf/Disease Found")
        plt.savefig(image_path)
        print("No leaf/Disease Found")
    keras.backend.clear_session()
