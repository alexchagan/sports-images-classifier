from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Rescaling, RandomRotation, RandomContrast, RandomZoom, BatchNormalization
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.efficientnet import EfficientNetB3
from keras import regularizers
from keras.optimizers import Adamax


# Preprocessing layers

rescale = Sequential()
rescale.add(Rescaling(1./255))
    

data_augment = Sequential()
data_augment.add(RandomRotation(0.2))
data_augment.add(RandomContrast(0.2))
data_augment.add(RandomZoom(0.2))


# Model definition

def custom_model(image_size, num_classes):

    '''A custom CNN architecture, consists of some convolution layers and max-pooling layers
    
    Parameters
    ----------
    image_size : int
        The size of image, height and width are same value
    num_class : int
        The number of classes for prediction

    Returns
    ----------
    model : Sequential
        A conifugred model ready for training

    '''

    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(image_size, image_size, 3)))
    model.add(data_augment)
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def vgg_model(num_classes):

    '''An architecture based on pretrained VGG16 CNN
    
    Parameters
    ----------
    num_class : int
        The number of classes for prediction

    Returns
    ----------
    model : Sequential
        A conifugred model ready for training

    '''
    
    vgg = VGG16(include_top = False, weights='imagenet')
    for layer in vgg.layers:
        layer.trainable = False

    model = Sequential()
    model.add(rescale)
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def enetb3_model(image_size, num_classes):
    
    '''An architecture based on pretrained EnetB3 CNN
    
    Parameters
    ----------
    image_size : int
        The size of image, height and width are same value
    num_class : int
        The number of classes for prediction

    Returns
    ----------
    model : Sequential
        A conifugred model ready for training

    '''
    # Normalization is included in the as part of the model, rescaling to [0,1] is not needed.
    b3 = EfficientNetB3(include_top=False, weights='imagenet',pooling='max', input_shape=(image_size,image_size,3))
    b3.trainable = True

    model = Sequential()
    model.add(b3)
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.006), activity_regularizer=regularizers.l1(0.006), bias_regularizer=regularizers.l1(0.006)))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adamax(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model