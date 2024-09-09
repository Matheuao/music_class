from model import *
from utils import *
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical


def image_and_class(path_transformation, resolution = [480,640]):

    imagens = []
    classes = []

    path0 = path_transformation + "/spec_original"
    i = 0
    for folder in os.listdir(path0):
        path1 = path0+"/"+folder
        print(path1)
        images, label = load_images_from_path(path1, i, resolution)
        imagens += images
        classes += label
        i += 1
    
    input_array = np.asarray(imagens)
    print(input_array.shape)

    path0 = path_transformation + "/spec_augmented"
    i = 0
    for folder in os.listdir(path0):
        path1 = path0+"/"+folder
        print(path1)
        images, label = load_images_from_path(path1, i, resolution)
        imagens += images
        classes += label
        i += 1
    
    input_array = np.asarray(imagens)
    print(input_array)
    
    #não gerou essa pasta antes, Matheus 
    #path0 = "../dataset/spec_mask_original"
    #i = 0
    #for folder in os.listdir(path0):
    #    path1 = path0+"/"+folder
    #    print(path1)
    #    images, label = load_images_from_path(path1, i)
    #    imagens += images
    #    classes += label
    #    i += 1
    #print(np.asarray(imagens).shape)

    path0 = path_transformation + "/spec_mask_augmented"
    i = 0
    for folder in os.listdir(path0):
        path1 = path0+"/"+folder
        print(path1)
        images, label = load_images_from_path(path1, i, resolution)
        imagens += images
        classes += label
        i += 1
    
    input_array = np.asarray(imagens)
    print(input_array.shape)

    return imagens, classes

def train_and_test_data(images, classes, test_len = 0.2):
    x_train, x_test, y_train, y_test = train_test_split(images, classes, stratify=classes, test_size= test_len, random_state=0)

    x_train_norm = np.array(x_train) / 255
    x_test_norm = np.array(x_test) / 255

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    return x_train_norm, x_test_norm, y_train_encoded, y_test_encoded


def train(path_transformation, batch, epochs,test_len = 0.2, resolution = [480,640]):

    images, classes =image_and_class(path_transformation, resolution=resolution)

    x_train, x_test, y_train, y_test = train_and_test_data(images, classes, test_len)

    model = music_classification_model(resolution)

    hist = model.fit(x_train, 
                     y_train, 
                    validation_data=(x_test, y_test), 
                    batch_size=batch, 
                    epochs=epochs)

    plot_performance(model,x_test,y_test,batch,hist)

    return model, hist

def save_model(model, path_save_model, path_save_wheight):
    
    # serialize model to JSON
    model_json = model.to_json()
    with open(path_save_model, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5 my_model.weights.h5
    model.save_weights(path_save_wheight)
    print("Saved model to disk")

def plot_performance(model,x_test_norm,y_test_encoded,batch,history):
    #def Validation_plot(history):
    print("Validation Accuracy",max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.show()

    #Validation_plot(hist)
    test_loss,test_acc=model.evaluate(x_test_norm,y_test_encoded,batch_size=batch)
    print("The test loss is ",test_loss)
    print("The best accuracy is: ",test_acc*100)

    
