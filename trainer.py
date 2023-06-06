from data_handler import download_data_to_local_directory
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

warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

num_classes = 100
IMAGE_SIZE = 224

print("Tensorflow is running on following devices : ")
print(device_lib.list_local_devices())

def prepare_dataset(visualize, batch_size):

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

    # IMPORTING DATASETS
    train_ds=image_dataset_from_directory("./sports-classifier-data/train",batch_size=batch_size,image_size=(IMAGE_SIZE,IMAGE_SIZE),seed=56)
    test_ds=image_dataset_from_directory("./sports-classifier-data/test",batch_size=batch_size,image_size=(IMAGE_SIZE,IMAGE_SIZE),seed=56)
    valid_ds=image_dataset_from_directory("./sports-classifier-data/valid",batch_size=batch_size,image_size=(IMAGE_SIZE,IMAGE_SIZE),seed=56)

    class_df = pd.read_csv('./sports-classifier-data/sports.csv')
    train_df = class_df[class_df['data set'] == 'train']
    class_distribution_print_and_csv(train_df, 'labels')

    class_names=train_ds.class_names
   
    if visualize:
    # HAVING A LOOK AT THE IMAGES OF TEST DATA
        plt.figure(figsize=(10,10))
        for images, labels in test_ds.take(1):
            for i in range(9):
                ax=plt.subplot(3, 3, i+1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.show()

        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break
  
    return train_ds, valid_ds, test_ds


def model_definition(num_classes, model_type):

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

    if model_type == 'custom':
        model = custom_model(image_size=IMAGE_SIZE, num_classes=num_classes)

           
    if model_type == 'vgg':
        model = vgg_model(image_size=IMAGE_SIZE, num_classes=num_classes)

        
    if model_type == "enetb3":
       model = enetb3_model(image_size=IMAGE_SIZE, num_classes=num_classes)

    return model

def trainer(model_type, num_classes, epochs, train_ds, valid_ds, gcp):

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
     
    model = model_definition(num_classes=num_classes, model_type=model_type)
    
    # SETTING UP CALLBACKS   
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
    parser.add_argument("--bucket_name", type=str, help="Bucket name on google cloud storage", default='sports-classifier-bucket')
    parser.add_argument("--batch_size", type=int, help="Batch size used by deep learning model", default=16)
    parser.add_argument("--epochs" , type=int, help="Number of epochs for training", default=40 )
    parser.add_argument("--download_data" , type=bool, help="Download dataset from google cloud storage bucket", default=False)
    parser.add_argument("--dummy_data" , type=bool, help="Download dummy data for testing", default=False)
    parser.add_argument("--gcp_training" , type=bool, help="Train on gcp vertex ai if true, train locally if false", default=False)
    args = parser.parse_args()


    if args.download_data:
        if args.dummy_data:
             download_data_to_local_directory('sports-classifier-bucket-dummy', './sports-classifier-data')
        else:
            download_data_to_local_directory(args.bucket_name, './sports-classifier-data')

    train_ds, valid_ds, test_ds = prepare_dataset(visualize=False, batch_size=args.batch_size)

    model_type = 'enetb3'

    trainer(model_type=model_type,num_classes=num_classes, train_ds=train_ds, valid_ds=valid_ds, epochs=args.epochs, gcp=args.gcp_training)

    model = keras.models.load_model('training_results/best.h5')

    predictor(model=model, test_ds=test_ds) 


