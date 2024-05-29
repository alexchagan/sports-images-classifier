import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from utils import makefolder
from metrics import Metrics

class Predictor:

    def __init__(self, test_ds, image_size, model_path):
        self._test_ds = test_ds
        self._image_size = image_size
        self._class_names_dict = {}
        self._model = tf.keras.models.load_model(model_path, custom_objects={"F1_score": Metrics.F1_score})

    def load_image(self, path, num_channels=3, interpolation='bilinear'):
        
        """Load an image from a path and resize it."""

        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
        img = tf.image.resize(img, self._image_size, method=interpolation)
        img.set_shape((self._image_size[0], self._image_size[1], num_channels))
        return img

    def create_class_to_name_dict(self):
        
        ''' Creates a dict with keys - lass integer values, values - names of classes '''

        classes = self._test_ds.class_names
        self._class_name_dict = {}
        index = 0
        for cl in classes:
            self._class_name_dict[index] = cl
            index += 1

    def predict_on_image(self, image_url):

        ''' Makes a prediction on a single image  '''

        image = self.load_image(path=image_url) 
        image = np.expand_dims(image, axis=0) # the model was trained on batches so we need to expand to (1,244,244,3)
        pred = self._model.predict(image)
        pred_class = np.argmax(pred)
        class_name = self._class_name_dict[pred_class]
        return class_name
    
    def predict(self):

        '''
        Predicts on the test dataset and compares the results to the ground truth.
        Writes results to a csv file and saves it in the inference_results folder.
        Creates and saves a confusion matrix in the inference_results folder. 
        '''

        makefolder('inference_results')

        y_true = []
        y_pred = []

        classes = self._test_ds.class_names
        class_count = len(classes)

        for image_batch, label_batch in self._test_ds:   
            # append true labels
            y_true.append(label_batch)
            # compute predictions
            preds = self._model.predict(image_batch)
            # append predicted labels
            y_pred.append(np.argmax(preds, axis = - 1))
        
        true_labels = tf.concat([item for item in y_true], axis = 0)
        predicted_labels = tf.concat([item for item in y_pred], axis = 0)

        _ = classification_report(true_labels, predicted_labels, target_names=classes, digits= 4)
        clr_csv = classification_report(true_labels, predicted_labels, target_names=classes, digits= 4, output_dict=True)

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

    
    

    
