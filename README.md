# sports-images-classifier
An image classification project on 100 classes of sport genres.
Uses a pretrained CNN EfficientNetB3 as base and fully connected layers for training.
Reaches a f1 score of 0.99 on test.
Dataset split is 12000,500,500 for train,val,test
link to dataset: https://www.kaggle.com/datasets/gpiosenka/sports-classification/download?datasetVersionNumber=8

--download_data parameter is false on default because it downloads from a googlecloud bucket
place test, train, valid folders from download link above into a folder called sports-classifier-data in root directory

train.py for training
inference.py for testing the model on test examples

