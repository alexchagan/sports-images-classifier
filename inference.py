import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from keras.utils import image_utils
from utilities import makefolder
import glob_vars as gv

def load_image(path, image_size=(224, 224), num_channels=3, interpolation='bilinear'):
  """Load an image from a path and resize it."""
  img = tf.io.read_file(path)
  img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
  img = tf.image.resize(img, image_size, method=interpolation)
  img.set_shape((image_size[0], image_size[1], num_channels))
  return img


def predictor(model, test_ds):

    '''Predicts on the test dataset and compares the results to the ground truth.
       Writes results to a csv file and saves it in the inference_results folder.
       Creates and saves a confusion matrix in the inference_results folder. 
    
    Parameters
    ----------
    model : Sequential
        A trained model  
    test_ds : tf.data.Dataset
        Test dataset
   
    '''

    y_true = []
    y_pred = []

    classes = test_ds.class_names
    class_count = len(classes)

    for image_batch, label_batch in test_ds:   # use dataset.unbatch() with repeat
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(image_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis = - 1))
    
    true_labels = tf.concat([item for item in y_true], axis = 0)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)

    makefolder('inference_results')

    clr = classification_report(true_labels, predicted_labels, target_names=classes, digits= 4)
    clr_csv = classification_report(true_labels, predicted_labels, target_names=classes, digits= 4, output_dict=True)
    print("Classification Report:\n----------------------\n", clr)
    report_df = pd.DataFrame(clr_csv).transpose()
    report_df.to_csv('inference_results/inference_report.csv')

    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
    plt.xticks(np.arange(class_count)+.5, classes, rotation=90)
    plt.yticks(np.arange(class_count)+.5, classes, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig('inference_results/matrix.jpg')

def create_class_to_name_dict(test_ds):
    
    '''Creates a dict with keys - lass integer values, values - names of classes  
    
    Parameters
    ---------- 
    test_ds : tf.data.Dataset
        Test dataset

    Returns
    ----------
    class_name_dict : dict
        Integer to class name dict

    '''
    classes = test_ds.class_names
    class_name_dict = {}
    index = 0
    for cl in classes:
        class_name_dict[index] = cl
        index += 1
    return class_name_dict

def predict_on_image(model, image_url, class_name_dict):

    '''Makes a prediction on a single image  
    
    Parameters
    ---------- 
    model : Sequential
        A trained model  
    image_url : string
        The path to the image
    class_name_dict : dict
        A dict with corresponding integers and class names

    Returns
    ----------
    class_name : string
        The name of the predicted class

    '''

    image = load_image(path=image_url) # preprocessing
    image = np.expand_dims(image, axis=0) # the model was trained on batches so we need to expand to (1,244,244,3)
    pred = model.predict(image)
    pred_class = np.argmax(pred)
    class_name = class_name_dict[pred_class]
    return class_name

if __name__ == '__main__':

    test_ds = image_dataset_from_directory("./sports_classifier_data/test", batch_size=gv.BATCH_SIZE, image_size=gv.IMAGE_SIZE, seed=56)
    model = keras.models.load_model('training_results/best.h5')
    class_name_dict = create_class_to_name_dict(test_ds)
    predictor(model, test_ds)
    
    

    
