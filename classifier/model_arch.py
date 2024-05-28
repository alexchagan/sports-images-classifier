import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.applications.efficientnet import EfficientNetB3
from keras import regularizers
from keras.optimizers import Adamax
from metrics import Metrics



class ModelArchitecture:

    def __init__(self, image_size, num_of_classes, model_name, learning_rate):

        self.image_size = image_size
        self.num_of_classes = num_of_classes
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.model = None

    # def set_enetb3_model(self):
        
    #     '''An architecture based on pretrained EnetB3 CNN
        
    #     Parameters
    #     ----------
    #     image_size : int
    #         The size of image, height and width are same value
    #     num_class : int
    #         The number of classes for prediction

    #     Returns
    #     ----------
    #     model : Sequential
    #         A conifugred model ready for training

    #     '''
    #     # Normalization is included in the as part of the model, rescaling to [0,1] is not needed.
    #     b3 = EfficientNetB3(include_top=False, weights='imagenet',pooling='max', input_shape=(self.image_size, self.image_size, 3))
    #     b3.trainable = True

    #     self.model = Sequential()
    #     self.model.add(b3)
    #     self.model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    #     self.model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.006), activity_regularizer=regularizers.l1(0.006), bias_regularizer=regularizers.l1(0.006)))
    #     self.model.add(Dropout(0.4))
    #     self.model.add(Dense(self.num_classes, activation='softmax'))
    #     self.model.compile(Adamax(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
   
    # def F1_score(y_true, y_pred): 

    #     '''define a function to compute the F1_score metric'''

    #     K = tf.keras.backend.backend()
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + K.epsilon())
    #     recall = true_positives / (possible_positives + K.epsilon())
    #     f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    #     return f1_val

    def make_model(self): 

        def F1_score(y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
            predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
            recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
            f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            return f1_val

        image_shape=(self.image_size[0], self.image_size[1], 3)

        if self.model_name == 'MobileNetV3Small':
            base_model=tf.keras.applications.MobileNetV3Small(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max')
            msg= 'created MobileNet V3  small model'

        elif self.model_name == 'MobileNetV3Large':
            base_model=tf.keras.applications.MobileNetV3Large(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max')
            msg='created MobileNetV3 large model'

        elif self.model_name == 'EfficientNetV2B0':        
            base_model=tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max')
            msg='Created EfficientNetV2 B0 model' 

        elif self.model_name == 'EfficientNetV2B1':
            base_model=tf.keras.applications.EfficientNetV2B1(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max') 
            msg='Created EfficientNetV2 B1 model'

        elif self.model_name == 'EfficientNetV2B2':        
            base_model=tf.keras.applications.EfficientNetV2B2(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max') 
            msg='Created EfficientNetV2 B2 model' 
        
        else:
            base_model = EfficientNetB3(include_top=False, weights='imagenet',pooling='max', input_shape=(self.image_size, self.image_size, 3))
            msg='Created EfficientNetV2 B3 model'

        base_model.trainable=True
        x=base_model.output
        x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
        x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                        bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
        x=Dropout(rate=.4, seed=123)(x)       
        output=Dense(self.num_of_classes, activation='softmax')(x)
        self.model=tf.keras.Model(inputs=base_model.input, outputs=output)
        self.model.compile(Adamax(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy', Metrics.F1_score]) 
        msg=msg + f' with initial learning rate set to {self.learning_rate}'
        print(msg)
        return self.model
    
   