from data_handler import download_from_kaggle
from utilities import tr_plot, class_distribution_print_and_csv, makefolder
from model_arch import custom_model, vgg_model, enetb3_model
from inference import predictor
from keras.utils import image_dataset_from_directory
from custom_callbacks import ASK
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime
from tensorflow.python.client import device_lib
import argparse
import os
from tensorflow import keras
import glob_vars as gv

warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

print("Tensorflow is running on following devices : ")
print(device_lib.list_local_devices())

def prepare_dataset(batch_size):

    '''Prepares the datasets for training and validation and prints class distribution  
    
    Parameters
    ----------
    visualize : boolean
        A flag for showing some examples from the dataset, images and their labels
    batch_size : int
        The batch size for model input, training and validation 

    Returns
    ----------
    train_ds, valid_ds, test_ds : tf.data.Dataset
        Dataset objects that yield batches of images with corresponding labels

    '''

    train_ds=image_dataset_from_directory("./sports_classifier_data/train",batch_size=batch_size,image_size=gv.IMAGE_SIZE,seed=56)
    test_ds=image_dataset_from_directory("./sports_classifier_data/test",batch_size=batch_size,image_size=gv.IMAGE_SIZE,seed=56)
    valid_ds=image_dataset_from_directory("./sports_classifier_data/valid",batch_size=batch_size,image_size=gv.IMAGE_SIZE,seed=56)

    class_df = pd.read_csv('./sports_classifier_data/sports.csv')
    train_df = class_df[class_df['data set'] == 'train']
    class_distribution_print_and_csv(train_df, 'labels')
    
    return train_ds, valid_ds, test_ds


def model_definition():

    '''Defines a compiled model architecture based on the stated model type   
    
    Parameters
    ----------
    num_classes : int
        Number of classes for prediction and size of the output layer
    model_type : string
        The name of the model architecture (custom, vgg, enetb3)

    Returns
    ----------
    model : Sequential
        A conifugred model ready for training

    '''

    if gv.MODEL_ARCH == 'vgg':
        return vgg_model(image_size=gv.IMAGE_SIZE[0], num_classes=gv.NUM_CLASSES)
    
    elif gv.MODEL_ARCH == "enetb3":
       return enetb3_model(image_size=gv.IMAGE_SIZE[0], num_classes=gv.NUM_CLASSES)

    else:
       return custom_model(image_size=gv.IMAGE_SIZE[0], num_classes=gv.NUM_CLASSES)

def trainer(epochs, train_ds, valid_ds, gcp):

    '''Trains a CNN model based on the requested architecture 
    
    Parameters
    ----------
    model_type : string
        The name of the model architecture (custom, vgg, enetb3)
    num_classes : int
        Number of classes for prediction and size of the output layer
    epochs : int
        Number of epochs for training
    train_ds : tf.data.Dataset
        Train dataset
    valid_ds : tf.data.Dataset
        Validation dataset
    gcp : boolean
        Flag for training on gcp platform

    '''
     
    model = model_definition()  

    log_dir = "tb_callback_dir/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if gcp:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'], histogram_freq=1) 
    else:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    makefolder('training_results')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='training_results/best.h5',  monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ask = ASK(model=model, epochs=epochs, ask_epoch=5)
    rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,verbose=1)
    estop=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, verbose=1,restore_best_weights=True)
    callbacks=[rlronp, estop, ask, tensorboard_callback, checkpoint_callback]

    history = model.fit(train_ds, validation_data=valid_ds, epochs=epochs, verbose=1, shuffle=False, callbacks=callbacks)
        
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    print(acc,val_acc,loss,val_loss)
    tr_plot(history,0)

    model.save('training_results/final.h5')
    print ('model was saved as: training_results/model.h5' )

   
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size used by deep learning model", default=16)
    parser.add_argument("--epochs" , type=int, help="Number of epochs for training", default=40 )
    parser.add_argument("--gcp_training" , type=bool, help="Train on gcp vertex ai if true, train locally if false", default=False)
    args = parser.parse_args()

    download_from_kaggle()

    train_ds, valid_ds, test_ds = prepare_dataset(batch_size=args.batch_size)

    trainer(train_ds=train_ds, valid_ds=valid_ds, epochs=args.epochs, gcp=args.gcp_training)

    model = keras.models.load_model('training_results/best.h5')

    predictor(model=model, test_ds=test_ds) 


